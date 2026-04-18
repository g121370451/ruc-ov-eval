import os
import sys
import yaml
import importlib 
from argparse import ArgumentParser
from src.core.logger import setup_logging
# ==========================================
# 1. 环境初始化
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR) # 代码仓库根目录
sys.path.append(REPO_ROOT)
WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)# 工作区根目录（Data和Output所在位置）
PROJECT_ROOT = WORKSPACE_ROOT

# 设置 OpenViking 配置文件路径（必须在 import openviking 之前）
ov_config_path = os.path.join(SCRIPT_DIR, "ov.conf")
if os.path.exists(ov_config_path):
    os.environ["OPENVIKING_CONFIG_FILE"] = ov_config_path
    print(f"[Init] Auto-detected OpenViking config: {ov_config_path}")

# 导入模块
try:
    from src.pipeline import BenchmarkPipeline
    from src.core.llm_client import LLMClientWrapper
except SyntaxError as e:
    print(f"\n[Fatal Error] 导入模块时发生语法错误: {e}")
    sys.exit(1)
except ImportError as e:
    print(f"\n[Fatal Error] 无法导入模块: {e}")
    print(f"当前 sys.path: {sys.path}\n")
    sys.exit(1)
# ==========================================
# 2. 辅助函数
# ==========================================

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def resolve_path(path_str, base_path):
    """
    将相对路径转换为基于 base_path 的绝对路径。如果 path_str 已经是绝对路径，则保持不变。
    """
    if not path_str:
        return path_str
    if os.path.isabs(path_str):
        return path_str
    # 规范化路径 (处理 ../ 等符号)
    return os.path.normpath(os.path.join(base_path, path_str))

# ==========================================
# 3. 主程序
# ==========================================

def main():
    parser = ArgumentParser(description="Run RAG Benchmark (Smart Path Handling)")
    # default_config_path = os.path.join(SCRIPT_DIR, "config/config.yaml")
    default_config_path = os.path.join(SCRIPT_DIR, "config_deepread/config.yaml")
    
    parser.add_argument("--config", default=default_config_path, 
                        help=f"Path to config file. Default: {default_config_path}")
    
    parser.add_argument("--step", choices=["all", "gen", "eval", "del"], default="all", 
                        help="Execution step: 'gen' (Retrieval+LLM), 'eval' (Judge), or 'all'")
    
    args = parser.parse_args()

    # --- B. 加载与解析 Config ---
    config_path = os.path.abspath(args.config)
    print(f"[Init] Loading configuration from: {config_path}")
    
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        return

    # --- C. 路径修正 ---
    print(f"[Init] Resolving paths relative to Project Root: {PROJECT_ROOT}")
    dataset_name = config.get('dataset_name', 'UnknownDataset')
    
    path_keys = ['raw_data', 'output_dir', 'vector_store', 'log_file', 'doc_output_dir']
    for key in path_keys:
        if key in config.get('paths', {}):
            original = config['paths'][key]
            rendered_path = original.format(dataset_name=dataset_name)
            resolved = resolve_path(rendered_path, PROJECT_ROOT)
            config['paths'][key] = resolved
            # print(f"  - {key}: {resolved}")

    # --- D. 初始化组件 ---
    try:
        logger = setup_logging(config['paths']['log_file'])
        logger.info(">>> Benchmark Session Started")
        
        # 1. Adapter (动态加载)
        adapter_cfg = config.get('adapter', {})
        module_path = adapter_cfg.get('module', 'src.adapters.locomo_adapter')
        class_name = adapter_cfg.get('class_name', 'LocomoAdapter')
        
        logger.info(f"Dynamically loading Adapter: {class_name} from {module_path}")
        logger.info(f"Loading raw data from: {config['paths']['raw_data']}")
        
        try:
            mod = importlib.import_module(module_path)
            AdapterClass = getattr(mod, class_name)
            adapter = AdapterClass(raw_file_path=config['paths']['raw_data'])
        except ImportError as e:
            logger.error(f"Could not import module '{module_path}'. Please check your config 'adapter.module'. Error: {e}")
            raise e
        except AttributeError as e:
            logger.error(f"Class '{class_name}' not found in module '{module_path}'. Please check your config 'adapter.class_name'. Error: {e}")
            raise e
        
        # 2. Vector Store（根据配置选择）
        store_cfg = config.get('store', {})
        store_type = store_cfg.get('type', 'viking')

        if store_type == 'DeepRead':
            from src.core.deepread_store import DeepReadWrapper
            vector_store = DeepReadWrapper.from_config(
                paths=config['paths'],
                llm_cfg=config.get('llm', {}),
                store_cfg=store_cfg
            )
        elif store_type == 'pageindex':
            from src.core.pageindex_store import PageIndexStoreWrapper
            pageindex_conf = store_cfg.get('pageindex_config_path')
            if pageindex_conf:
                pageindex_conf = resolve_path(pageindex_conf, PROJECT_ROOT)
            vector_store = PageIndexStoreWrapper(
                store_path=config['paths']['vector_store'],
                doc_output_dir=config['paths'].get('doc_output_dir', ''),
                config_path=pageindex_conf
            )
        else:
            from src.core.vector_store import VikingStoreWrapper
            vector_store = VikingStoreWrapper(store_path=config['paths']['vector_store'])
        
        # 3. LLM Client
        api_key = os.environ.get(
            config['llm'].get('api_key_env_var', ''), 
            config['llm'].get('api_key')
        )
        if not api_key:
            logger.warning("No API Key found in config or environment variables!")
            
        llm_client = LLMClientWrapper(config=config['llm'], api_key=api_key)

        # 4. Pipeline
        pipeline = BenchmarkPipeline(
            config=config,
            adapter=adapter,
            vector_db=vector_store,
            llm=llm_client
        )

        # --- E. 执行任务 ---
        if args.step in ["all", "gen"]:
            logger.info("Stage: Generation (Ingest -> Retrieve -> Generate)")
            pipeline.run_generation()
            
        if args.step in ["all", "eval"]:
            logger.info("Stage: Evaluation (Judge -> Metrics)")
            pipeline.run_evaluation()

        if args.step in ["del"]:
            logger.info("Stage: Delete Vector Store")
            pipeline.run_deletion()
        
        logger.info("Benchmark finished successfully.")

    except KeyboardInterrupt:
        print("\n[Stop] Execution interrupted by user.")
    except Exception as e:
        if 'logger' in locals():
            logger.exception("Fatal error during execution")
        print(f"\n[Fatal Error] 程序运行出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
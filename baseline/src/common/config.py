import yaml
import argparse
from pathlib import Path
from typing import Dict, Any
from copy import deepcopy


class Config:
    def __init__(self):
        self.data = {}

    def load_yaml(self, path: str) -> None:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        # 处理科学记数法和类型转换
        for key, value in config.items():
            if isinstance(value, str):
                # 尝试转换为float或int
                try:
                    # 先尝试float（支持科学记数法如5e-4）
                    if 'e' in value.lower() or '.' in value:
                        value = float(value)
                    else:
                        # 尝试int
                        value = int(value)
                except (ValueError, AttributeError):
                    # 保持原值
                    pass
            config[key] = value

        # 处理权重扰动子配置
        # 如果 perturbation 被指定（不是 'none'），自动加载对应的子配置
        if 'perturbation' in config and config['perturbation'] != 'none':
            perturb_type = config['perturbation']
            if perturb_type in config and isinstance(config[perturb_type], dict):
                perturb_config = config[perturb_type]
                # 将子配置的参数添加到主配置中（带前缀）
                for sub_key, sub_value in perturb_config.items():
                    config[f"{perturb_type}_{sub_key}"] = sub_value

        self.data.update(config)

    def load_from_args(self, args: argparse.Namespace) -> None:
        # 获取项目根目录（src的父目录）
        src_dir = Path(__file__).parent.parent
        project_dir = src_dir.parent

        dataset_config = project_dir / f'configs/datasets/{args.dataset}.yaml'
        if dataset_config.exists():
            self.load_yaml(str(dataset_config))

        # 模型配置通过 get_model() 函数直接处理，无需加载 YAML 文件
        # 只需通过 args.model 指定模型名称

        train_config = project_dir / f'configs/training/{args.config}.yaml'
        if train_config.exists():
            self.load_yaml(str(train_config))

        self._apply_overrides(args)

    def _apply_overrides(self, args: argparse.Namespace) -> None:
        # 先添加必要的参数（dataset, model, config）
        if args.dataset:
            self.data['dataset'] = args.dataset
        if args.model:
            self.data['model'] = args.model
        if args.config:
            self.data['config'] = args.config

        # 然后添加其他非None的参数
        for key, value in vars(args).items():
            if value is None or key in ['dataset', 'model', 'config']:
                continue
            self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getattr__(self, key: str) -> Any:
        if key.startswith('_'):
            return super().__getattribute__(key)
        return self.data.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def to_dict(self) -> Dict[str, Any]:
        return deepcopy(self.data)

    def __repr__(self) -> str:
        return f"Config({self.data})"

"""
handles loading, validation, and management of the sandbox runner configuration
YAML file configuration

Configuration hierarchy:
- YAML file (base)
- Environment variables (override)
- CLI arguments (final override)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os

import yaml


@dataclass
class RunnerConfig:
    """Runner identity and metadata configuration."""
    id: str = "runner-1"
    name: Optional[str] = None
    location: Optional[str] = None


@dataclass
class APIConfig:
    """API Gateway configuration."""
    port: int = 8080
    require_api_key: bool = True
    rate_limit_per_validator: int = 1  # requests per minute
    api_key_header: str = "X-API-Key"


@dataclass
class HardwareConfig:
    """Hardware resources configuration."""
    gpu_count: int = 8
    gpu_type: str = "H200"
    cpu_cores: int = 96
    memory_gb: int = 1024


@dataclass
class ExecutionConfig:
    """Job execution configuration."""
    mode: str = "docker+gvisor"  # docker+gvisor | docker | direct
    fallback_on_error: bool = True
    cpu_limit: int = 32
    memory_limit_gb: int = 256
    prep_timeout_seconds: int = 7200  # 2 hours
    inference_timeout_seconds: int = 3600  # 1 hour
    disk_quota_gb: int = 100
    max_processes: int = 128
    
    # repository settings
    repo_clone_timeout_seconds: int = 600  # 10 minutes
    repo_build_timeout_seconds: int = 3600  # 1 hour
    allowed_repo_hosts: list[str] = field(default_factory=lambda: ["github.com", "gitlab.com"])
    
    # log settings
    show_terminal_logs: bool = True 
    persist_logs: bool = True
    log_retention_hours: int = 1


@dataclass
class GVisorConfig:
    """gVisor security runtime configuration."""
    enabled: bool = True
    platform: str = "ptrace"  # ptrace | kvm
    network_mode: str = "none"  # none | host | sandbox
    file_access: str = "exclusive"  # exclusive | shared
    overlay: bool = True


@dataclass
class SeccompConfig:
    """Seccomp syscall filtering configuration."""
    enabled: bool = True
    profile_path: Optional[Path] = None
    default_action: str = "SCMP_ACT_ERRNO"
    allowed_syscalls: list[str] = field(default_factory=list)


@dataclass
class NetworkPolicyConfig:
    """Network access policy configuration."""
    prep_allow_internet: bool = True
    inference_block_internet: bool = True
    allowed_prep_domains: list[str] = field(default_factory=lambda: [
        "huggingface.co",
        "github.com",
        "gitlab.com",
        "raw.githubusercontent.com"
    ])
    blocked_domains: list[str] = field(default_factory=list)


@dataclass
class SecurityConfig:
    """Security layers configuration."""
    gvisor: GVisorConfig = field(default_factory=GVisorConfig)
    seccomp: SeccompConfig = field(default_factory=SeccompConfig)
    network_policy: NetworkPolicyConfig = field(default_factory=NetworkPolicyConfig)
    apparmor_profile: Optional[str] = None
    selinux_context: Optional[str] = None
    readonly_rootfs: bool = True
    no_new_privileges: bool = True
    drop_capabilities: list[str] = field(default_factory=lambda: [
        "CAP_SYS_ADMIN",
        "CAP_NET_ADMIN",
        "CAP_SYS_MODULE",
    ])


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    prometheus_port: int = 9090
    prometheus_path: str = "/metrics"
    enable_falco: bool = True
    falco_rules_path: Optional[Path] = None
    log_format: str = "json"  # json | text
    log_file: Optional[Path] = None


@dataclass
class StorageConfig:
    """S3 storage configuration."""
    s3_endpoint: Optional[str] = None
    s3_bucket: str = "hone-subnet-data"
    s3_region: str = "us-east-1"
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    input_prefix: str = "inputs/"
    output_prefix: str = "outputs/"


@dataclass
class Config:
    """Root configuration object containing all subsections."""
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    api: APIConfig = field(default_factory=APIConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


def load_config(config_path: Path) -> Config:
    """
    Load configuration from YAML file with environment variable overrides.
    
    Environment variables can override config values:
    - SHOW_TERMINAL_LOGS: true/false to enable/disable terminal log display
    - LOG_RETENTION_HOURS: number of hours to keep logs
    - PERSIST_LOGS: true/false to enable/disable log persistence
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with loaded and validated settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
        ValueError: If configuration is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    if not yaml_data:
        raise ValueError("Configuration file is empty")
    
    config = Config()
    
    if 'runner' in yaml_data:
        config.runner = RunnerConfig(**yaml_data['runner'])
    
    if 'api' in yaml_data:
        api_data = yaml_data['api'].copy()
        config.api = APIConfig(**api_data)
    
    if 'hardware' in yaml_data:
        config.hardware = HardwareConfig(**yaml_data['hardware'])
    
    if 'execution' in yaml_data:
        exec_data = yaml_data['execution'].copy()
        
        # Apply environment variable overrides for log settings
        if 'SHOW_TERMINAL_LOGS' in os.environ:
            exec_data['show_terminal_logs'] = os.environ['SHOW_TERMINAL_LOGS'].lower() == 'true'
        
        if 'PERSIST_LOGS' in os.environ:
            exec_data['persist_logs'] = os.environ['PERSIST_LOGS'].lower() == 'true'
        
        if 'LOG_RETENTION_HOURS' in os.environ:
            try:
                exec_data['log_retention_hours'] = int(os.environ['LOG_RETENTION_HOURS'])
            except ValueError:
                pass  # Keep default if invalid
        
        config.execution = ExecutionConfig(**exec_data)
    
    if 'security' in yaml_data:
        security_data = yaml_data['security']
        
        if 'gvisor' in security_data:
            config.security.gvisor = GVisorConfig(**security_data['gvisor'])
        
        if 'seccomp' in security_data:
            seccomp_data = security_data['seccomp'].copy()
            if 'profile_path' in seccomp_data and seccomp_data['profile_path']:
                seccomp_data['profile_path'] = Path(seccomp_data['profile_path'])
            config.security.seccomp = SeccompConfig(**seccomp_data)
        
        if 'network_policy' in security_data:
            config.security.network_policy = NetworkPolicyConfig(**security_data['network_policy'])
        
        for key in ['apparmor_profile', 'selinux_context', 'readonly_rootfs', 
                    'no_new_privileges', 'drop_capabilities']:
            if key in security_data:
                setattr(config.security, key, security_data[key])
    
    if 'monitoring' in yaml_data:
        monitoring_data = yaml_data['monitoring'].copy()
        if 'falco_rules_path' in monitoring_data and monitoring_data['falco_rules_path']:
            monitoring_data['falco_rules_path'] = Path(monitoring_data['falco_rules_path'])
        if 'log_file' in monitoring_data and monitoring_data['log_file']:
            monitoring_data['log_file'] = Path(monitoring_data['log_file'])
        config.monitoring = MonitoringConfig(**monitoring_data)
    
    if 'storage' in yaml_data:
        config.storage = StorageConfig(**yaml_data['storage'])
    
    _validate_config(config)
    
    return config


def _validate_config(config: Config):
    """
    Validate configuration values for consistency and correctness.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    valid_modes = ["docker+gvisor", "docker", "direct"]
    if config.execution.mode not in valid_modes:
        raise ValueError(f"Invalid execution mode: {config.execution.mode}. Must be one of {valid_modes}")
    
    if config.hardware.gpu_count < 1:
        raise ValueError(f"GPU count must be at least 1, got {config.hardware.gpu_count}")
    
    if not (1024 <= config.api.port <= 65535):
        raise ValueError(f"API port must be between 1024 and 65535, got {config.api.port}")
    
    if config.execution.cpu_limit < 1:
        raise ValueError(f"CPU limit must be at least 1, got {config.execution.cpu_limit}")
    
    if config.execution.memory_limit_gb < 1:
        raise ValueError(f"Memory limit must be at least 1GB, got {config.execution.memory_limit_gb}")
    
    if config.execution.prep_timeout_seconds < 60:
        raise ValueError(f"Prep timeout must be at least 60 seconds, got {config.execution.prep_timeout_seconds}")
    
    if config.execution.inference_timeout_seconds < 60:
        raise ValueError(f"Inference timeout must be at least 60 seconds, got {config.execution.inference_timeout_seconds}")
    
    if config.execution.log_retention_hours < 0:
        raise ValueError(f"Log retention must be at least 0 hours, got {config.execution.log_retention_hours}")

def generate_default_config(output_path: Path):
    """
    Generate a default configuration YAML file.
    
    Args:
        output_path: Path where to write the default config
    """
    default_config = {
        'runner': {
            'id': 'runner-1',
            'name': 'Default Sandbox Runner',
            'location': 'us-east-1'
        },
        'api': {
            'port': 8443,
            'require_api_key': True,
            'rate_limit_per_validator': 3,
        },
        'hardware': {
            'gpu_count': 1,  # testing phase
            'gpu_type': 'H200',
            'cpu_cores': 15,
            'memory_gb': 175
        },
        'execution': {
            'mode': 'docker',
            'fallback_on_error': True,
            'cpu_limit': 5,  # testing phase
            'memory_limit_gb': 100,
            'prep_timeout_seconds': 3600,
            'inference_timeout_seconds': 3600,
            'disk_quota_gb': 100,
            'max_processes': 32,
            'repo_clone_timeout_seconds': 600,
            'repo_build_timeout_seconds': 1800,
            'show_terminal_logs': True,
            'persist_logs': True,
            'log_retention_hours': 1
        },
        'security': {
            'seccomp': {
                'enabled': True,
                'default_action': 'SCMP_ACT_ERRNO'
            },
            'network_policy': {
                'prep_allow_internet': True,
                'inference_block_internet': True,
                'allowed_prep_domains': [
                    'huggingface.co',
                    'github.com',
                    'gitlab.com'
                ]
            },
            'readonly_rootfs': True,
            'no_new_privileges': True,
            'drop_capabilities': [
                'CAP_SYS_ADMIN',
                'CAP_NET_ADMIN',
                'CAP_SYS_MODULE'
            ]
        },
        'monitoring': {
            'prometheus_port': 9090,
            'prometheus_path': '/metrics',
            'enable_falco': True,
            'log_format': 'json'
        },
        'storage': {
            's3_bucket': 'hone-subnet-data',
            's3_region': 'us-east-1',
            'input_prefix': 'inputs/',
            'output_prefix': 'outputs/'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
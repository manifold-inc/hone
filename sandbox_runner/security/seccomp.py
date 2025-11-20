"""
Seccomp Syscall Filtering

Provides system call filtering using seccomp (secure computing mode):
- Whitelist/blacklist specific system calls
- Reduce attack surface by limiting available syscalls
- Prevent privilege escalation attempts
- Log suspicious syscall attempts

Seccomp profiles can be applied to Docker containers and direct processes.

Security modes:
- SCMP_ACT_ALLOW: Allow syscall
- SCMP_ACT_ERRNO: Return error for syscall
- SCMP_ACT_KILL: Kill process on syscall
- SCMP_ACT_LOG: Log syscall attempt
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class SeccompError(Exception):
    """Raised when seccomp operation fails"""
    pass


class SeccompProfile:
    """
    Represents a seccomp filter profile
    
    A profile defines which system calls are allowed/denied
    and what action to take for denied calls
    """
    
    def __init__(
        self,
        default_action: str = "SCMP_ACT_ERRNO",
        allowed_syscalls: Optional[List[str]] = None,
        blocked_syscalls: Optional[List[str]] = None
    ):
        """
        Initialize seccomp profile
        
        Args:
            default_action: Default action for syscalls
            allowed_syscalls: List of explicitly allowed syscalls
            blocked_syscalls: List of explicitly blocked syscalls
        """
        self.default_action = default_action
        self.allowed_syscalls = allowed_syscalls or []
        self.blocked_syscalls = blocked_syscalls or []
    
    def to_docker_seccomp(self) -> Dict:
        """
        Convert profile to Docker seccomp format
        
        Returns:
            Dictionary in Docker seccomp JSON format
        """
        profile = {
            "defaultAction": self.default_action,
            "architectures": [
                "SCMP_ARCH_X86_64",
                "SCMP_ARCH_X86",
                "SCMP_ARCH_X32"
            ],
            "syscalls": []
        }
        
        if self.allowed_syscalls:
            profile["syscalls"].append({
                "names": self.allowed_syscalls,
                "action": "SCMP_ACT_ALLOW"
            })
        
        if self.blocked_syscalls:
            profile["syscalls"].append({
                "names": self.blocked_syscalls,
                "action": "SCMP_ACT_ERRNO"
            })
        
        return profile
    
    def save_to_file(self, path: Path):
        """
        Save profile to JSON file
        
        Args:
            path: Path to save profile
        """
        profile_dict = self.to_docker_seccomp()
        
        with open(path, 'w') as f:
            json.dump(profile_dict, f, indent=2)
        
        logger.info(f"Saved seccomp profile to {path}")


class SeccompManager:
    """
    Manager for seccomp syscall filtering
    
    Provides high-level interface for creating and managing seccomp profiles
    for sandboxed execution
    """
    
    DANGEROUS_SYSCALLS = [
        "init_module",
        "finit_module",
        "delete_module",
        
        "mount",
        "umount",
        "umount2",
        "pivot_root",
        
        "settimeofday",
        "clock_settime",
        
        "reboot",
        
        "iopl",
        "ioperm",
        
        "add_key",
        "request_key",
        "keyctl",
        
        "bpf",
        
        "perf_event_open",
        
        "unshare",
        "setns",
    ]
    
    ESSENTIAL_SYSCALLS = [
        "read",
        "write",
        "open",
        "openat",
        "close",
        "stat",
        "fstat",
        "lstat",
        "access",
        "faccessat",
        "readlink",
        "readlinkat",
        
        "getdents",
        "getdents64",
        "mkdir",
        "mkdirat",
        "rmdir",
        
        "mmap",
        "munmap",
        "mprotect",
        "brk",
        "mremap",
        
        "clone",
        "fork",
        "vfork",
        "execve",
        "execveat",
        "exit",
        "exit_group",
        "wait4",
        "waitid",
        
        "rt_sigaction",
        "rt_sigprocmask",
        "rt_sigreturn",
        "sigaltstack",
        
        "gettimeofday",
        "clock_gettime",
        "clock_getres",
        "nanosleep",
        
        "getpid",
        "getuid",
        "getgid",
        "geteuid",
        "getegid",
        "getppid",
        "prctl",
        "arch_prctl",
        "futex",
        "set_tid_address",
        "set_robust_list",
        "get_robust_list",
        "sched_yield",
        "sched_getaffinity",
        "sched_setaffinity",
    ]
    KNOWN_SYSCALLS = {
        "read", "write", "open", "close", "stat", "fstat", "lstat", "poll",
        "lseek", "mmap", "mprotect", "munmap", "brk", "rt_sigaction",
        "rt_sigprocmask", "rt_sigreturn", "ioctl", "pread64", "pwrite64",
        "readv", "writev", "access", "pipe", "select", "sched_yield",
        "mremap", "msync", "mincore", "madvise", "shmget", "shmat", "shmctl",
        "dup", "dup2", "pause", "nanosleep", "getitimer", "alarm", "setitimer",
        "getpid", "sendfile", "socket", "connect", "accept", "sendto", "recvfrom",
        "sendmsg", "recvmsg", "shutdown", "bind", "listen", "getsockname",
        "getpeername", "socketpair", "setsockopt", "getsockopt", "clone", "fork",
        "vfork", "execve", "exit", "wait4", "kill", "uname", "semget", "semop",
        "semctl", "shmdt", "msgget", "msgsnd", "msgrcv", "msgctl", "fcntl",
        "flock", "fsync", "fdatasync", "truncate", "ftruncate", "getdents",
        "getcwd", "chdir", "fchdir", "rename", "mkdir", "rmdir", "creat", "link",
        "unlink", "symlink", "readlink", "chmod", "fchmod", "chown", "fchown",
        "lchown", "umask", "gettimeofday", "getrlimit", "getrusage", "sysinfo",
        "times", "ptrace", "getuid", "syslog", "getgid", "setuid", "setgid",
        "geteuid", "getegid", "setpgid", "getppid", "getpgrp", "setsid",
        "setreuid", "setregid", "getgroups", "setgroups", "setresuid",
        "getresuid", "setresgid", "getresgid", "getpgid", "setfsuid", "setfsgid",
        "getsid", "capget", "capset", "rt_sigpending", "rt_sigtimedwait",
        "rt_sigqueueinfo", "rt_sigsuspend", "sigaltstack", "utime", "mknod",
        "uselib", "personality", "ustat", "statfs", "fstatfs", "sysfs",
        "getpriority", "setpriority", "sched_setparam", "sched_getparam",
        "sched_setscheduler", "sched_getscheduler", "sched_get_priority_max",
        "sched_get_priority_min", "sched_rr_get_interval", "mlock", "munlock",
        "mlockall", "munlockall", "vhangup", "modify_ldt", "pivot_root",
        "_sysctl", "prctl", "arch_prctl", "adjtimex", "setrlimit", "chroot",
        "sync", "acct", "settimeofday", "mount", "umount2", "swapon", "swapoff",
        "reboot", "sethostname", "setdomainname", "iopl", "ioperm",
        "create_module", "init_module", "delete_module", "get_kernel_syms",
        "query_module", "quotactl", "nfsservctl", "getpmsg", "putpmsg", "afs_syscall",
        "tuxcall", "security", "gettid", "readahead", "setxattr", "lsetxattr",
        "fsetxattr", "getxattr", "lgetxattr", "fgetxattr", "listxattr",
        "llistxattr", "flistxattr", "removexattr", "lremovexattr", "fremovexattr",
        "tkill", "time", "futex", "sched_setaffinity", "sched_getaffinity",
        "set_thread_area", "io_setup", "io_destroy", "io_getevents", "io_submit",
        "io_cancel", "get_thread_area", "lookup_dcookie", "epoll_create",
        "epoll_ctl_old", "epoll_wait_old", "remap_file_pages", "getdents64",
        "set_tid_address", "restart_syscall", "semtimedop", "fadvise64",
        "timer_create", "timer_settime", "timer_gettime", "timer_getoverrun",
        "timer_delete", "clock_settime", "clock_gettime", "clock_getres",
        "clock_nanosleep", "exit_group", "epoll_wait", "epoll_ctl", "tgkill",
        "utimes", "vserver", "mbind", "set_mempolicy", "get_mempolicy",
        "mq_open", "mq_unlink", "mq_timedsend", "mq_timedreceive", "mq_notify",
        "mq_getsetattr", "kexec_load", "waitid", "add_key", "request_key",
        "keyctl", "ioprio_set", "ioprio_get", "inotify_init", "inotify_add_watch",
        "inotify_rm_watch", "migrate_pages", "openat", "mkdirat", "mknodat",
        "fchownat", "futimesat", "newfstatat", "unlinkat", "renameat", "linkat",
        "symlinkat", "readlinkat", "fchmodat", "faccessat", "pselect6", "ppoll",
        "unshare", "set_robust_list", "get_robust_list", "splice", "tee",
        "sync_file_range", "vmsplice", "move_pages", "utimensat", "epoll_pwait",
        "signalfd", "timerfd_create", "eventfd", "fallocate", "timerfd_settime",
        "timerfd_gettime", "accept4", "signalfd4", "eventfd2", "epoll_create1",
        "dup3", "pipe2", "inotify_init1", "preadv", "pwritev", "rt_tgsigqueueinfo",
        "perf_event_open", "recvmmsg", "fanotify_init", "fanotify_mark",
        "prlimit64", "name_to_handle_at", "open_by_handle_at", "clock_adjtime",
        "syncfs", "sendmmsg", "setns", "getcpu", "process_vm_readv",
        "process_vm_writev", "kcmp", "finit_module", "sched_setattr",
        "sched_getattr", "renameat2", "seccomp", "getrandom", "memfd_create",
        "kexec_file_load", "bpf", "execveat", "userfaultfd", "membarrier",
        "mlock2", "copy_file_range", "preadv2", "pwritev2", "pkey_mlock",
        "pkey_alloc", "pkey_free", "statx", "io_pgetevents", "rseq"
    }
    
    def __init__(self):
        """Initialize seccomp manager"""
        self.profiles: Dict[str, SeccompProfile] = {}
        
        self._create_default_profiles()
        
        logger.info("Seccomp manager initialized")
    
    def _create_default_profiles(self):
        """Create default seccomp profiles"""
        
        self.profiles['strict'] = SeccompProfile(
            default_action="SCMP_ACT_ERRNO",
            allowed_syscalls=self.ESSENTIAL_SYSCALLS
        )
        
        moderate_allowed = self.ESSENTIAL_SYSCALLS + [
            "socket",
            "connect",
            "bind",
            "listen",
            "accept",
            "accept4",
            "sendto",
            "recvfrom",
            "sendmsg",
            "recvmsg",
            "shutdown",
            "getsockname",
            "getpeername",
            "socketpair",
            "setsockopt",
            "getsockopt",
            "ioctl",
            "fcntl",
            "poll",
            "select",
            "pselect6",
            "epoll_create",
            "epoll_create1",
            "epoll_ctl",
            "epoll_wait",
            "epoll_pwait",
        ]
        
        self.profiles['moderate'] = SeccompProfile(
            default_action="SCMP_ACT_ERRNO",
            allowed_syscalls=moderate_allowed,
            blocked_syscalls=self.DANGEROUS_SYSCALLS
        )
        
        self.profiles['permissive'] = SeccompProfile(
            default_action="SCMP_ACT_ALLOW",
            blocked_syscalls=self.DANGEROUS_SYSCALLS
        )
        
        logger.info(f"Created {len(self.profiles)} default seccomp profiles")
    
    def get_profile(self, name: str) -> Optional[SeccompProfile]:
        """
        Get a seccomp profile by name
        
        Args:
            name: Profile name ('strict', 'moderate', 'permissive')
            
        Returns:
            SeccompProfile object or None
        """
        return self.profiles.get(name)
    
    def create_custom_profile(
        self,
        name: str,
        default_action: str = "SCMP_ACT_ERRNO",
        allowed_syscalls: Optional[List[str]] = None,
        blocked_syscalls: Optional[List[str]] = None
    ) -> SeccompProfile:
        """
        Create a custom seccomp profile
        
        Args:
            name: Profile name
            default_action: Default action for syscalls
            allowed_syscalls: List of allowed syscalls
            blocked_syscalls: List of blocked syscalls
            
        Returns:
            SeccompProfile object
        """
        profile = SeccompProfile(
            default_action=default_action,
            allowed_syscalls=allowed_syscalls,
            blocked_syscalls=blocked_syscalls
        )
        
        self.profiles[name] = profile
        
        logger.info(f"Created custom seccomp profile: {name}")
        
        return profile
    
    def load_profile_from_file(self, path: Path, name: str) -> SeccompProfile:
        """
        Load seccomp profile from JSON file
        
        Args:
            path: Path to JSON profile file
            name: Name to assign to profile
            
        Returns:
            SeccompProfile object
            
        Raises:
            SeccompError: If file cannot be loaded
        """
        try:
            with open(path, 'r') as f:
                profile_dict = json.load(f)
            
            allowed_syscalls = []
            blocked_syscalls = []
            
            for syscall_rule in profile_dict.get('syscalls', []):
                action = syscall_rule.get('action')
                names = syscall_rule.get('names', [])
                
                if action == 'SCMP_ACT_ALLOW':
                    allowed_syscalls.extend(names)
                elif action == 'SCMP_ACT_ERRNO':
                    blocked_syscalls.extend(names)
            
            profile = SeccompProfile(
                default_action=profile_dict.get('defaultAction', 'SCMP_ACT_ERRNO'),
                allowed_syscalls=allowed_syscalls,
                blocked_syscalls=blocked_syscalls
            )
            
            self.profiles[name] = profile
            
            logger.info(f"Loaded seccomp profile from {path}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Failed to load seccomp profile: {e}")
            raise SeccompError(f"Failed to load profile: {e}")
    
    def save_profile(self, name: str, path: Path):
        """
        Save a seccomp profile to file
        
        Args:
            name: Profile name
            path: Path to save profile
            
        Raises:
            SeccompError: If profile not found or save fails
        """
        profile = self.profiles.get(name)
        if not profile:
            raise SeccompError(f"Profile not found: {name}")
        
        try:
            profile.save_to_file(path)
        except Exception as e:
            raise SeccompError(f"Failed to save profile: {e}")
    
    def get_docker_seccomp_config(self, profile_name: str) -> Optional[Dict]:
        """
        Get Docker seccomp configuration for a profile
        
        Args:
            profile_name: Name of profile to use
            
        Returns:
            Docker seccomp configuration dictionary or None
        """
        profile = self.get_profile(profile_name)
        if not profile:
            logger.warning(f"Profile not found: {profile_name}")
            return None
        
        return profile.to_docker_seccomp()
    
    def is_syscall_allowed(
        self,
        profile_name: str,
        syscall_name: str
    ) -> bool:
        """
        Check if a syscall is allowed by a profile
        
        Args:
            profile_name: Profile name
            syscall_name: Syscall to check
            
        Returns:
            True if syscall is allowed
        """
        profile = self.get_profile(profile_name)
        if not profile:
            return False
        
        if syscall_name in profile.blocked_syscalls:
            return False
        
        if syscall_name in profile.allowed_syscalls:
            return True
        
        return profile.default_action == "SCMP_ACT_ALLOW"
    
    def get_profile_summary(self, profile_name: str) -> Dict:
        """
        Get a summary of a seccomp profile
        
        Args:
            profile_name: Profile name
            
        Returns:
            Dictionary with profile summary
        """
        profile = self.get_profile(profile_name)
        if not profile:
            return {"error": "Profile not found"}
        
        return {
            "name": profile_name,
            "default_action": profile.default_action,
            "allowed_syscalls_count": len(profile.allowed_syscalls),
            "blocked_syscalls_count": len(profile.blocked_syscalls),
            "allowed_syscalls": profile.allowed_syscalls[:10],  
            "blocked_syscalls": profile.blocked_syscalls[:10],
        }
    
    def list_profiles(self) -> List[str]:
        """
        List available seccomp profiles
        
        Returns:
            List of profile names
        """
        return list(self.profiles.keys())
    
    def validate_syscall_list(self, syscalls: List[str]) -> Dict:
        """
        Validate a list of syscalls against the Linux syscall table
        
        Args:
            syscalls: List of syscall names to validate
            
        Returns:
            Dictionary with validation results
        """
        valid = []
        unknown = []
        dangerous = []
        
        for syscall in syscalls:
            if syscall in self.DANGEROUS_SYSCALLS:
                dangerous.append(syscall)
            elif syscall in self.KNOWN_SYSCALLS:
                valid.append(syscall)
            else:
                unknown.append(syscall)
        
        return {
            "valid": valid,
            "unknown": unknown,
            "dangerous": dangerous,
            "total": len(syscalls),
            "validation_summary": {
                "valid_count": len(valid),
                "unknown_count": len(unknown),
                "dangerous_count": len(dangerous)
            }
        }
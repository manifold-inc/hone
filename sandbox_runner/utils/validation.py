"""
Repository Validation Module

Validates miner repositories before execution:
- Check for required files (inference.py, requirements.txt)
- Validate requirements.txt for malicious packages
- Check repository structure
- Validate Dockerfile if present
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when repository validation fails."""
    pass


class RepositoryValidator:
    """
    Validates miner repository structure and contents.
    
    Ensures repositories meet minimum requirements and don't contain
    obvious security issues.
    """
    
    # Packages that are blacklisted due to security concerns
    BLACKLISTED_PACKAGES = {
    }
    
    # Packages that require review (suspicious but not always malicious)
    SUSPICIOUS_PACKAGES = {
        'eval',
        'exec',
        'compile',
        '__import__',
        'subprocess',
        'os.system',
    }
    
    # Required files in repository
    REQUIRED_FILES = [
        'inference.py',
        'requirements.txt'
    ]
    
    # Optional files
    OPTIONAL_FILES = [
        'Dockerfile',
        'README.md',
        '.gitignore'
    ]
    
    # Maximum file sizes (in bytes)
    MAX_REQUIREMENTS_SIZE = 1024 * 100  # 100 KB
    MAX_INFERENCE_SIZE = 1024 * 1024 * 10  # 10 MB
    MAX_DOCKERFILE_SIZE = 1024 * 100  # 100 KB
    
    def __init__(self, allowed_hosts: Optional[List[str]] = None):
        """
        Initialize repository validator.
        
        Args:
            allowed_hosts: List of allowed repository hosts (e.g., github.com)
        """
        self.allowed_hosts = allowed_hosts or ['github.com', 'gitlab.com']
        
    def validate_structure(self, repo_path: Path) -> bool:
        """
        Validate repository structure.
        
        Checks:
        1. Repository directory exists
        2. Required files are present
        3. Files are not suspiciously large
        
        Args:
            repo_path: Path to repository directory
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not repo_path.exists():
            raise ValidationError(f"Repository directory not found: {repo_path}")
        
        if not repo_path.is_dir():
            raise ValidationError(f"Repository path is not a directory: {repo_path}")
        
        # Check for required files
        for required_file in self.REQUIRED_FILES:
            file_path = repo_path / required_file
            if not file_path.exists():
                raise ValidationError(
                    f"Required file missing: {required_file}. "
                    f"Miner repositories must include: {', '.join(self.REQUIRED_FILES)}"
                )
            
            if not file_path.is_file():
                raise ValidationError(f"Required path is not a file: {required_file}")
        
        # Check file sizes
        self._check_file_size(
            repo_path / 'inference.py',
            self.MAX_INFERENCE_SIZE,
            'inference.py'
        )
        
        self._check_file_size(
            repo_path / 'requirements.txt',
            self.MAX_REQUIREMENTS_SIZE,
            'requirements.txt'
        )
        
        # Check Dockerfile if present
        dockerfile_path = repo_path / 'Dockerfile'
        if dockerfile_path.exists():
            self._check_file_size(
                dockerfile_path,
                self.MAX_DOCKERFILE_SIZE,
                'Dockerfile'
            )
        
        logger.info(f"Repository structure validation passed: {repo_path}")
        return True
    
    def validate_requirements(self, requirements_path: Path) -> bool:
        """
        Validate requirements.txt for malicious or problematic packages.
        
        Checks:
        1. File is readable and well-formed
        2. No blacklisted packages
        3. Warn about suspicious packages
        4. No excessive dependencies (> 100)
        
        Args:
            requirements_path: Path to requirements.txt
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not requirements_path.exists():
            raise ValidationError(f"requirements.txt not found: {requirements_path}")
        
        try:
            with open(requirements_path, 'r') as f:
                content = f.read()
        except Exception as e:
            raise ValidationError(f"Failed to read requirements.txt: {e}")
        
        # Parse requirements
        packages = self._parse_requirements(content)
        
        # Check for excessive dependencies
        if len(packages) > 100:
            raise ValidationError(
                f"Too many dependencies ({len(packages)}). Maximum is 100. "
                "This may indicate a malicious or poorly configured repository."
            )
        
        # Check for blacklisted packages
        blacklisted_found = []
        for package in packages:
            package_name = package.split('==')[0].split('>=')[0].split('<=')[0].strip()
            if package_name.lower() in self.BLACKLISTED_PACKAGES:
                blacklisted_found.append(package_name)
        
        if blacklisted_found:
            raise ValidationError(
                f"Blacklisted packages found: {', '.join(blacklisted_found)}. "
                "These packages pose security risks and are not allowed."
            )
        
        # Warn about suspicious packages (but don't fail)
        suspicious_found = []
        for package in packages:
            package_name = package.split('==')[0].split('>=')[0].split('<=')[0].strip()
            if package_name.lower() in self.SUSPICIOUS_PACKAGES:
                suspicious_found.append(package_name)
        
        if suspicious_found:
            logger.warning(
                f"Suspicious packages found in requirements.txt: {', '.join(suspicious_found)}. "
                "These will be monitored during execution."
            )
        
        logger.info(
            f"Requirements validation passed: {len(packages)} packages",
            extra={"package_count": len(packages)}
        )
        return True
    
    def validate_inference_script(self, inference_path: Path) -> bool:
        """
        Validate inference.py script.
        
        Basic checks:
        1. File is valid Python
        2. No obvious shell command execution
        3. Has required argument parsing
        
        Args:
            inference_path: Path to inference.py
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not inference_path.exists():
            raise ValidationError(f"inference.py not found: {inference_path}")
        
        try:
            with open(inference_path, 'r') as f:
                content = f.read()
        except Exception as e:
            raise ValidationError(f"Failed to read inference.py: {e}")
        
        # Basic syntax check - try to compile
        try:
            compile(content, str(inference_path), 'exec')
        except SyntaxError as e:
            raise ValidationError(f"Invalid Python syntax in inference.py: {e}")
        
        # Check for dangerous patterns (basic checks only)
        dangerous_patterns = [
            r'os\.system\(',
            r'subprocess\.call\(',
            r'subprocess\.run\(',
            r'eval\(',
            r'exec\(',
            r'__import__\(',
        ]
        
        found_dangerous = []
        for pattern in dangerous_patterns:
            if re.search(pattern, content):
                found_dangerous.append(pattern)
        
        if found_dangerous:
            logger.warning(
                f"Potentially dangerous patterns found in inference.py: {found_dangerous}. "
                "These will be monitored during execution."
            )
        
        # Check for required argument parsing (--phase, --input, --output)
        # This is a soft check - we'll handle it gracefully if missing
        has_phase_arg = '--phase' in content or 'argparse' in content
        if not has_phase_arg:
            logger.warning(
                "inference.py may not have proper argument parsing for --phase. "
                "Ensure the script accepts: --phase prep|inference --input PATH --output PATH"
            )
        
        logger.info("Inference script validation passed")
        return True
    
    def validate_dockerfile(self, dockerfile_path: Path) -> bool:
        """
        Validate Dockerfile if present.
        
        Basic checks:
        1. File is readable
        2. Has FROM instruction
        3. No obviously malicious commands
        
        Args:
            dockerfile_path: Path to Dockerfile
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not dockerfile_path.exists():
            # Dockerfile is optional
            return True
        
        try:
            with open(dockerfile_path, 'r') as f:
                content = f.read()
        except Exception as e:
            raise ValidationError(f"Failed to read Dockerfile: {e}")
        
        # Check for FROM instruction
        if not re.search(r'^FROM\s+', content, re.MULTILINE):
            raise ValidationError(
                "Dockerfile must have a FROM instruction"
            )
        
        # Check for dangerous commands
        dangerous_patterns = [
            r'rm\s+-rf\s+/',  # Dangerous deletion
            r'chmod\s+777',  # Overly permissive
            r'curl.*\|\s*bash',  # Pipe to bash
            r'wget.*\|\s*sh',  # Pipe to shell
        ]
        
        found_dangerous = []
        for pattern in dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                found_dangerous.append(pattern)
        
        if found_dangerous:
            logger.warning(
                f"Potentially dangerous commands found in Dockerfile: {found_dangerous}"
            )
        
        logger.info("Dockerfile validation passed")
        return True
    
    def validate_url(self, repo_url: str) -> bool:
        """
        Validate repository URL.
        
        Checks:
        1. URL is from allowed host
        2. URL format is correct
        
        Args:
            repo_url: Repository URL
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Check if URL is from allowed host
        is_allowed = any(host in repo_url for host in self.allowed_hosts)
        
        if not is_allowed:
            raise ValidationError(
                f"Repository URL not from allowed host: {repo_url}. "
                f"Allowed hosts: {', '.join(self.allowed_hosts)}"
            )
        
        # Basic URL format check
        if not (repo_url.startswith('https://') or repo_url.startswith('git@')):
            raise ValidationError(
                f"Invalid repository URL format: {repo_url}. "
                "Must start with https:// or git@"
            )
        
        logger.info(f"Repository URL validation passed: {repo_url}")
        return True
    
    def validate_all(self, repo_path: Path, repo_url: str) -> bool:
        """
        Run all validation checks.
        
        Args:
            repo_path: Path to cloned repository
            repo_url: Original repository URL
            
        Returns:
            True if all validations pass
            
        Raises:
            ValidationError: If any validation fails
        """
        # Validate URL
        self.validate_url(repo_url)
        
        # Validate structure
        self.validate_structure(repo_path)
        
        # Validate requirements.txt
        self.validate_requirements(repo_path / 'requirements.txt')
        
        # Validate inference.py
        self.validate_inference_script(repo_path / 'inference.py')
        
        # Validate Dockerfile if present
        self.validate_dockerfile(repo_path / 'Dockerfile')
        
        logger.info(f"All validation checks passed for {repo_url}")
        return True
    
    def _parse_requirements(self, content: str) -> List[str]:
        """
        Parse requirements.txt content into list of packages.
        
        Args:
            content: requirements.txt content
            
        Returns:
            List of package specifications
        """
        packages = []
        
        for line in content.split('\n'):
            # Strip whitespace and comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Skip pip options
            if line.startswith('-'):
                continue
            
            packages.append(line)
        
        return packages
    
    def _check_file_size(self, file_path: Path, max_size: int, name: str):
        """
        Check if file size is within limits.
        
        Args:
            file_path: Path to file
            max_size: Maximum allowed size in bytes
            name: File name for error messages
            
        Raises:
            ValidationError: If file exceeds size limit
        """
        if file_path.exists():
            size = file_path.stat().st_size
            if size > max_size:
                raise ValidationError(
                    f"{name} is too large: {size} bytes (max {max_size} bytes)"
                )
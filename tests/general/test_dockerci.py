import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestDockerCI(unittest.TestCase):

    @staticmethod
    def _write_executable(path, content):
        path.write_text(content)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def test_container_name_includes_workflow_identity(self):
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docker_log = temp_path / 'docker.log'
            script = (repo_root / '.dev_scripts/dockerci.sh').read_text()
            portable_script = script.replace('exec {lock_fd}>"/tmp/gpu$gpu" || exit 1', 'lock_fd=9')
            self.assertNotEqual(script, portable_script)
            script_path = temp_path / 'dockerci.sh'
            self._write_executable(script_path, portable_script)
            self._write_executable(
                temp_path / 'docker',
                '#!/bin/sh\n'
                'printf "%s\\n" "$*" >> "$DOCKER_LOG"\n',
            )
            self._write_executable(temp_path / 'flock', '#!/bin/sh\nexit 0\n')

            env = os.environ.copy()
            env.update({
                'PATH': f'{temp_path}:{env["PATH"]}',
                'DOCKER_LOG': str(docker_log),
                'GITHUB_RUN_ID': '12345',
                'GITHUB_RUN_ATTEMPT': '2',
                'IMAGE_NAME': 'swift-ci-image',
                'IMAGE_VERSION': 'latest',
                'MODELSCOPE_CACHE': str(temp_path / 'modelscope-cache'),
                'MODELSCOPE_HOME_CACHE': str(temp_path / 'home-cache'),
                'CI_COMMAND': 'true',
            })

            result = subprocess.run(
                ['bash', script_path],
                cwd=repo_root,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            docker_commands = docker_log.read_text().splitlines()
            run_command = next(command for command in docker_commands if command.startswith('run '))
            self.assertIn('--name swift-ci-12345-2-0', run_command)


if __name__ == '__main__':
    unittest.main()

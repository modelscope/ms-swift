# Contributor Guide

_Welcome to offer PRs, bug reports, documentation supplements or other types of contributions to SWIFT!_

## Table of Contents
- [Code of Conduct](#-code-of-conduct)
- [Contribution Process](#-contribution-process)
- [Hardware support](#-Hardware-support)

## 📖 Code of Conduct
Please refer to our [Code of Conduct documentation](./CODE_OF_CONDUCT.md).

## 🔁 Contribution Process
### What We Need
- New Technologies and New Models: SWIFT needs to support more open-source models and datasets, or new technologies that we have not paid attention to. If you are interested please submit a PR to us.
- Technical Propagation: If you are interested in technical propagation, you are welcome to help us write tutorials, documents or videos on any website, and send us the link.
- Community Contribution: You can write technical articles related to SWIFT, and submit them to us. After review and approval, we will publish them on the official ModelScope accounts (Zhihu, WeChat, etc.), with your name assigned.

### Incentives
- we will issue electronic certificates to contributors on behalf of the ModelScope community, to encourage your selfless contributions.
- We will offer small souvenirs related to the ModelScope Community.
- We will provide free A10 computing power during the development period. For more details, please refer to [Hardware-support](#-Hardware-support) section.

### Submitting PR (Pull Requests)

Any feature development is carried out in the form of Fork and then PR on GitHub.
1. Fork: Go to the [ms-swift](https://github.com/modelscope/ms-swift) page and click the **Fork button**. After completion, a SWIFT code repository will be cloned under your personal organization.
2. Clone: Clone the code repository generated in the first step to your local machine and **create a new branch** for development. During development, please click the **Sync Fork button** in time to synchronize with the `main` branch to prevent code expiration and conflicts.
3. Submit PR: After development and testing, push the code to the remote branch. On GitHub, go to the **Pull Requests page**, create a new PR, select your code branch as the source branch, and the `modelscope/swift:main` branch as the target branch.

4. Write Description: It is necessary to provide a good feature description in the PR, so that the reviewers know the content of your modification.
5. Review: We hope that the code to be merged is concise and efficient, so we may raise some questions and discuss them. Please note that any issues raised in the review are aimed at the code itself, not at you personally. Once all issues are discussed and resolved, your code will be approved.

### Code Standards and Development Approach
SWIFT has conventional variable naming conventions and development approaches. Please follow these approaches as much as possible during development.
1. Variable names are separated by underscores, and class names are named with the first letter of each word capitalized.
2. All Python indentation uses four spaces instead of a tab.
3. Choose well-known open-source libraries, avoid using closed-source libraries or unstable open-source libraries, and avoid repeating the existing code.

After the PR is submitted, SWIFT will perform two types of tests:
- Code Lint Test: A static code compliance check test. please make sure that you have performed code lint locally in advance.
```shell
pip install pre-commit # In the swift folder
pre-commit run --all-files # Fix the errors reported by pre-commit until all checks are successful
```
- CI Tests: Smoke tests and unit tests, please refer to the next section.

### Running CI Tests
Before submitting the PR, please ensure that your development code is protected by test cases, such as smoke tests for new features, or unit tests for various edge cases. Reviewers will also pay attention to this during code review. At the same time, there will be dedicated services running CI Tests, running all test cases, and the code can only be merged after the test cases pass.

Additionally, since some important tests have been skipped due to long running time, to ensure that your logic is correct, you can run the test locally:
```shell
python tests/llm/test_run.py
```
Please make sure this test can pass normally.

## ✅ Hardware support

SWIFT will provide hardware support for developers, including free GPUs. If needed, please email us ([contact@modelscope.cn](mailto:contact@modelscope.cn)) or join our WeChat group:

<p align="left">
<img src="asset/wechat.png" width="250" style="display: inline-block;">
</p>

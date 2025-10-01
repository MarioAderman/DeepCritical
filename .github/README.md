# GitHub Repository Configuration

This directory contains the complete GitHub repository configuration for DeepCritical, including issue templates, workflows, and settings.

## 📁 Directory Structure

```
.github/
├── ISSUE_TEMPLATE/           # Issue templates
│   ├── bug_report.yml       # Bug report template
│   ├── feature_request.yml  # Feature request template
│   ├── documentation.yml    # Documentation issue template
│   ├── performance.yml      # Performance issue template
│   ├── question.yml         # Question template
│   ├── bioinformatics.yml   # Bioinformatics-specific template
│   └── config.yml           # Issue template configuration
├── workflows/               # GitHub Actions workflows
│   ├── ci.yml              # Continuous Integration
│   ├── release.yml         # Release management
│   └── dependabot.yml      # Dependabot auto-merge
├── repository-settings.yml  # Repository settings documentation
├── SECURITY.md             # Security policy
└── README.md               # This file
```

## 🎯 Issue Templates

### Bug Report (`bug_report.yml`)
- Comprehensive bug reporting with component selection
- Environment details and reproduction steps
- Error logs and configuration information
- Severity classification and impact assessment

### Feature Request (`feature_request.yml`)
- Detailed feature specification with use cases
- Component and feature type classification
- Implementation notes and priority assessment
- Target audience and expected outcomes

### Documentation (`documentation.yml`)
- Documentation improvement requests
- Issue type classification (missing, outdated, unclear)
- Affected sections and target audience
- Examples and desired state specification

### Performance (`performance.yml`)
- Performance issue reporting with metrics
- Component and performance type classification
- Profiling data and workload details
- Expected vs. actual performance comparison

### Question (`question.yml`)
- Usage and configuration questions
- Context and attempted solutions
- Environment details and expected outcomes
- Component-specific guidance

### Bioinformatics (`bioinformatics.yml`)
- Domain-specific bioinformatics issues
- Data source and biological context
- Scientific impact and research implications
- Specialized configuration and results

## 🔄 GitHub Actions Workflows

### Continuous Integration (`ci.yml`)
- **Lint**: Code style and formatting checks
- **Test**: Unit and integration tests across Python versions
- **Integration Test**: End-to-end functionality testing
- **Security**: Security scanning with Bandit
- **Build**: Package building and artifact generation

### Release Management (`release.yml`)
- **Release**: Automated release creation with changelog
- **PyPI Publishing**: Package publishing to PyPI
- **Conda Publishing**: Conda package publishing (planned)

### Dependabot (`dependabot.yml`)
- **Auto-merge**: Automated dependency updates
- **Testing**: Validation of dependency updates
- **Squash Merge**: Clean commit history

## ⚙️ Repository Settings

### Branch Protection
- Required status checks for main branch
- Pull request reviews required
- Stale review dismissal
- Force push prevention

### Issue Labels
- **Priority**: Critical, High, Medium, Low
- **Type**: Bug, Enhancement, Documentation, Performance, Question
- **Component**: Core, PRIME, Bioinformatics, DeepSearch, Challenge, Tools, Agents, Config, Graph, Docs
- **Status**: Needs Triage, In Progress, Blocked, Needs Review, Ready for Testing, Resolved

### Security Features
- Vulnerability alerts enabled
- Secret scanning active
- Dependabot security updates
- Private vulnerability reporting

## 🔒 Security Configuration

### Security Policy (`SECURITY.md`)
- Supported versions and response timeline
- Vulnerability reporting process
- Security considerations and best practices
- Responsible disclosure guidelines

### Security Features
- Private vulnerability reporting
- Automated security scanning
- Dependency vulnerability monitoring
- Secure configuration management

## 📋 Dependabot Configuration

### Automated Updates
- **Python Dependencies**: Weekly updates with version constraints
- **GitHub Actions**: Weekly updates for workflow dependencies
- **Docker**: Weekly updates for container dependencies

### Update Management
- Auto-merge for minor/patch updates
- Manual review for major updates
- Comprehensive testing before merge

## 🚀 Getting Started

### For Contributors
1. Review [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines
2. Use appropriate issue templates for bug reports and feature requests
3. Follow the pull request template for code contributions
4. Ensure all CI checks pass before requesting review

### For Maintainers
1. Review and apply repository settings from `repository-settings.yml`
2. Configure branch protection rules
3. Set up required secrets for CI/CD workflows
4. Monitor and respond to security reports

### For Users
1. Use issue templates for bug reports and questions
2. Check existing issues before creating new ones
3. Provide detailed information for better support
4. Follow the code of conduct

## 🔧 Configuration Setup

### Required Secrets
Set these secrets in your repository settings:

```bash
# PyPI Publishing
PYPI_TOKEN=your_pypi_token

# API Keys (for testing)
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
SERPER_API_KEY=your_serper_key

# Coverage Reporting
CODECOV_TOKEN=your_codecov_token
```

### Branch Protection Rules
Configure branch protection for the main branch:
- Require status checks: `ci`, `lint`, `test`
- Require pull request reviews: 1 approval
- Dismiss stale reviews: enabled
- Require up-to-date branches: enabled

### Issue Labels
Import the labels from `repository-settings.yml`:
- Priority labels (Critical, High, Medium, Low)
- Type labels (Bug, Enhancement, Documentation, etc.)
- Component labels (Core, PRIME, Bioinformatics, etc.)
- Status labels (Needs Triage, In Progress, etc.)

## 📊 Monitoring and Analytics

### CI/CD Metrics
- Build success rates
- Test coverage trends
- Security scan results
- Performance benchmarks

### Issue Management
- Issue resolution times
- Component-specific metrics
- Priority distribution
- Contributor activity

### Release Management
- Release frequency
- Version adoption
- Bug fix turnaround
- Feature delivery

## 🔄 Maintenance

### Regular Tasks
- Review and update dependency versions
- Monitor security advisories
- Update issue templates as needed
- Review and improve workflows

### Quarterly Reviews
- Analyze issue and PR metrics
- Review security policy effectiveness
- Update contribution guidelines
- Assess workflow performance

## 📚 Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Issue Templates Guide](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests)
- [Security Best Practices](https://docs.github.com/en/code-security)
- [Dependabot Documentation](https://docs.github.com/en/code-security/dependabot)

## 🤝 Contributing

To improve the GitHub configuration:

1. Create an issue using the appropriate template
2. Make changes in a feature branch
3. Test changes in a fork or test repository
4. Submit a pull request with detailed description
5. Ensure all CI checks pass

## 📄 License

This configuration is part of the DeepCritical project and is licensed under the MIT License.

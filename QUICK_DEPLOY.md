# 快速部署清单

## 📋 部署前检查

- [ ] 已有 GitHub 账号和空仓库
- [ ] 已有 PyPI 账号
- [ ] 已安装 git
- [ ] 已安装 Python 3.8+

## 🚀 快速部署命令

### 第一步：上传到 GitHub

```bash
# 1. 初始化 Git
cd /Volumes/Yangdong/causal-narrative
git init
git add .
git commit -m "Initial commit: causal-narrative v0.1.0"

# 2. 连接远程仓库（替换为您的仓库地址）
git remote add origin https://github.com/causalis-nlp/causal-narrative.git

# 3. 推送到 GitHub
git branch -M main
git push -u origin main
```

### 第二步：构建并上传到 PyPI

```bash
# 1. 安装工具
pip install --upgrade pip setuptools wheel twine build

# 2. 清理并构建
rm -rf build/ dist/ *.egg-info
python -m build

# 3. 检查包
twine check dist/*

# 4. 上传到 PyPI（需要先配置 .pypirc 或使用 --username --password）
twine upload dist/*
```

### 第三步：创建 GitHub Release

```bash
# 1. 创建标签
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0

# 2. 在 GitHub 网页上创建 Release
# 访问：https://github.com/causalis-nlp/causal-narrative/releases/new
```

## 🔑 配置 PyPI 凭证

创建 `~/.pypirc` 文件：

```bash
cat > ~/.pypirc << 'EOF'
[distutils]
index-servers =
    pypi

[pypi]
username = __token__
password = pypi-your-api-token-here
EOF

chmod 600 ~/.pypirc
```

## ✅ 验证部署

```bash
# 测试从 PyPI 安装
pip install causal-narrative

# 测试导入
python -c "import causal_narrative; print(causal_narrative.__version__)"
```

## 📝 需要更新的信息

在部署前，请确认以下文件中的信息：

1. **pyproject.toml** - 已更新 ✅
   - 作者：causalis-nlp
   - 邮箱：causalisnlp@gmail.com
   - GitHub URL：需要确认实际仓库地址

2. **README.md** - 需要更新
   - 将所有 `your-username` 替换为实际的 GitHub 用户名
   - 更新 citation 中的作者信息

3. **CHANGELOG.md** - 已创建 ✅

4. **.gitignore** - 已创建 ✅

5. **MANIFEST.in** - 已创建 ✅

## 🔄 更新版本流程

```bash
# 1. 修改版本号
vim causal_narrative/__init__.py  # 更新 __version__

# 2. 更新 CHANGELOG
vim CHANGELOG.md

# 3. 提交并推送
git add .
git commit -m "Release v0.1.1"
git push

# 4. 重新构建并上传
rm -rf dist/ build/ *.egg-info
python -m build
twine upload dist/*

# 5. 创建新标签
git tag -a v0.1.1 -m "Release version 0.1.1"
git push origin v0.1.1
```

## 📞 获取帮助

- 详细指南：查看 `DEPLOYMENT_GUIDE.md`
- PyPI 文档：https://pypi.org/help/
- GitHub 文档：https://docs.github.com/

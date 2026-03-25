# 中文支持更新总结

## 🎉 已完成的工作

### 1. 添加 HanLP SRL 支持

**修改的文件：**
- `causal_narrative/semantic_role_labeling.py`
  - 添加了 `HanLPSRL` 类用于中文语义角色标注
  - 更新 `get_srl()` 工厂函数支持 'hanlp' 方法
  - 添加 `is_hanlp_available()` 检查函数

**功能：**
```python
from causal_narrative import get_srl

# 使用 HanLP 进行中文 SRL
srl = get_srl('hanlp')
result = srl.process("政府提高了利率。")
```

### 2. 添加中文 BERT Embedding 支持

**修改的文件：**
- `causal_narrative/embedding.py`
  - 添加 `DEFAULT_CHINESE_MODEL_NAME` 常量
  - 添加 `detect_language()` 函数自动检测语言
  - 添加 `get_default_model_for_language()` 函数
  - 更新 `load_embedder()` 支持语言参数

**功能：**
```python
from causal_narrative.embedding import SentenceEmbedder, DEFAULT_CHINESE_MODEL_NAME

# 使用中文 embedding 模型
embedder = SentenceEmbedder(model_name=DEFAULT_CHINESE_MODEL_NAME)

# 或者自动检测
from causal_narrative.embedding import load_embedder
embedder = load_embedder(language='zh')
```

### 3. 创建中文教程 Notebook

**新文件：**
- `notebook/tutorial_minimal_zh.ipynb`

**内容：**
- 完整的中文因果叙事分析流程
- 中文例句示例
- HanLP SRL 使用示例
- 中文 BERT embedding 和聚类
- 中文因果网络可视化

### 4. 更新 README 文档

**修改的文件：**
- `README.md`

**更新内容：**
- 在 SRL 部分添加了中文示例
- 添加 "Language Support" 部分
- 新增 "Option 3: Chinese Language Support" 安装指南
- 添加中文使用示例
- 指向中文教程 notebook

### 5. 更新包配置和版本

**修改的文件：**
- `causal_narrative/__init__.py` - 版本号更新为 `0.2.0`
- `pyproject.toml` - 添加 `chinese` 可选依赖组
- `CHANGELOG.md` - 添加 v0.2.0 更新日志

**新的可选依赖：**
```bash
# 安装中文支持
pip install 'causal-narrative[chinese]'

# 或安装所有功能
pip install 'causal-narrative[all]'
```

## 📦 如何上传到 GitHub 和 PyPI

### 步骤 1: 提交代码到 GitHub

```bash
# 查看修改
git status

# 添加所有修改
git add .

# 提交
git commit -m "Add Chinese language support (v0.2.0)

- Add HanLP SRL for Chinese text
- Add Chinese BERT embedding with auto-detection
- Add Chinese tutorial notebook
- Update documentation with Chinese examples
- Version bump to 0.2.0"

# 推送到 GitHub
git push origin main
```

### 步骤 2: 创建 Git Tag

```bash
# 创建版本标签
git tag -a v0.2.0 -m "Release v0.2.0: Chinese language support"

# 推送标签（这会触发 GitHub Actions 自动发布到 PyPI）
git push origin v0.2.0
```

### 步骤 3: GitHub Actions 自动发布

已经配置好的 `.github/workflows/publish.yml` 会自动：
1. 检测到 v0.2.0 标签
2. 构建包
3. 上传到 PyPI

**前提条件：**
- 需要在 GitHub Secrets 中配置 `PYPI_API_TOKEN`
  - 访问：https://github.com/causal-narrative/causal-narrative/settings/secrets/actions
  - 名称：`PYPI_API_TOKEN`
  - 值：您的 PyPI API token

### 步骤 4: 手动发布（如果 GitHub Actions 失败）

```bash
# 清理旧的构建
rm -rf dist/ build/ *.egg-info

# 构建包
python -m build

# 检查包
twine check dist/*

# 上传到 PyPI
twine upload dist/*
```

## 🧪 测试新功能

### 测试中文 SRL

```python
from causal_narrative import get_srl, is_hanlp_available

# 检查 HanLP 是否可用
print(f"HanLP available: {is_hanlp_available()}")

# 使用 HanLP
if is_hanlp_available():
    srl = get_srl('hanlp')
    result = srl.process("政府提高了利率。")
    print(result)
```

### 测试中文 Embedding

```python
from causal_narrative.embedding import (
    SentenceEmbedder,
    detect_language,
    DEFAULT_CHINESE_MODEL_NAME
)

# 检测语言
lang = detect_language("这是一个中文句子")
print(f"Detected language: {lang}")

# 使用中文模型
embedder = SentenceEmbedder(model_name=DEFAULT_CHINESE_MODEL_NAME)
embeddings = embedder.embed(["中文句子1", "中文句子2"])
print(f"Embeddings shape: {embeddings.shape}")
```

### 运行中文教程

```bash
# 启动 Jupyter
jupyter notebook notebook/tutorial_minimal_zh.ipynb
```

## 📝 发布检查清单

- [x] 添加 HanLP SRL 支持
- [x] 添加中文 BERT embedding 支持
- [x] 创建中文教程 notebook
- [x] 更新 README 文档
- [x] 更新版本号到 0.2.0
- [x] 更新 CHANGELOG
- [x] 更新 pyproject.toml 依赖配置
- [ ] 提交代码到 GitHub
- [ ] 创建并推送 v0.2.0 标签
- [ ] 验证 GitHub Actions 发布成功
- [ ] 测试从 PyPI 安装新版本

## 🎯 下一步

1. **推送到 GitHub:**
   ```bash
   git add .
   git commit -m "Add Chinese language support (v0.2.0)"
   git push origin main
   ```

2. **创建 Release:**
   ```bash
   git tag -a v0.2.0 -m "Release v0.2.0: Chinese language support"
   git push origin v0.2.0
   ```

3. **验证发布:**
   - 检查 GitHub Actions: https://github.com/causal-narrative/causal-narrative/actions
   - 检查 PyPI: https://pypi.org/project/causal-narrative/

4. **测试安装:**
   ```bash
   pip install --upgrade causal-narrative
   python -c "import causal_narrative; print(causal_narrative.__version__)"
   # 应该输出: 0.2.0
   ```

## 🌟 新功能亮点

- **🇨🇳 完整的中文支持**：从 SRL 到聚类的完整流程
- **🤖 智能语言检测**：自动选择合适的模型
- **📚 详细教程**：中文 notebook 示例
- **🔧 灵活配置**：可选依赖，按需安装

## 📞 如果遇到问题

1. **GitHub 推送失败**：检查网络，使用 SSH 或代理
2. **PyPI 上传失败**：确保 API Token 已配置
3. **HanLP 安装问题**：使用 `pip install hanlp` 单独安装
4. **模型下载慢**：首次运行时需要下载模型，请耐心等待

祝发布顺利！🚀

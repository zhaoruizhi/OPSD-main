# Git 回滚操作指南

这个项目后续建议用 git 管住每次实验改动。这样 vLLM probe、HF probe、prompt 模版这类切换都可以用分支和提交精确回退。

## 初始化基线

在服务器项目目录执行：

```bash
cd /home/ruizzhao/OPSD-main
git init
git add .
git commit -m "baseline"
```

## 大改前新建分支

```bash
git checkout -b experiment/vllm-logit-probe
```

## 查看当前改动

```bash
git status
git diff
```

## 只撤销某个文件

```bash
git checkout -- eval/quick_logit_probe.py
```

## 撤销所有未提交改动

这个命令会丢弃当前工作区所有未提交修改，执行前先确认没有要保留的结果文件或代码：

```bash
git reset --hard
```

## 已提交后回滚

先看提交历史：

```bash
git log --oneline
```

用反向提交撤销某次改动：

```bash
git revert <commit_hash>
```

## 保存检查点

如果只是想先存一个可以回来的状态：

```bash
git add .
git commit -m "checkpoint: before changing logit probe backend"
```

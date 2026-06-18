## 基本操作
查看全局git配置
git config --global --list
查看当前仓库配置
git config --list
git config --local --list
查看当前所处分支以及未提交/未跟踪的修改
git status
查看本地及远程所有分支
git branch -a
查看远程仓库地址
git remote -v
查看本地及远程所有提交历史
git log
git log -5
git log --oneline

查看未提交的修改
git log origin/main..HEAD --oneline
输出 01d8c91 (HEAD -> main) 0603修改1
查看commit当中改了哪些文件，每个文件增删多少行
git show --stat 01d8c91
只显示改了哪些文件，不显示具体内容
git show --name-only 01d8c91

## 从上游更新
### 1. 先更新main分支（从上游）
- git checkout main
- git fetch upstream
- git merge upstream/main
- git reset --hard upstream/main (抛弃本地修改)
- git push origin main

### 2. 再更新dev分支（从main）
- git checkout dev
- git merge main
- git push origin dev

## 本地开发
### 在dev分支开发
- git checkout dev
- 开发代码...
- 提交修改到dev
-- git add .
-- git commit -m "feat: 完成xxx功能"
-- git push origin dev

## 将dev的修改合并到main
### 先切换回main并更新（确保main是最新的）
git checkout main
git pull origin main
### 将dev合并到main
git merge dev
### 推送更新后的main
git push origin main
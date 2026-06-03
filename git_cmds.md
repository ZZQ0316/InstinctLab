### 从上游更新
# 1. 先更新main分支（从上游）
git checkout main
git fetch upstream
git merge upstream/main
// git reset --hard upstream/main (抛弃本地修改)
git push origin main

# 2. 再更新dev分支（从main）
git checkout dev
git merge main
git push origin dev

### 本地开发
# 1. 在dev分支开发
git checkout dev
# 开发代码...

# 2. 提交修改到dev
git add .
git commit -m "feat: 完成xxx功能"
git push origin dev

### 将dev的修改合并到main
# 3. 先切换回main并更新（确保main是最新的）
git checkout main
git pull origin main

# 4. 将dev合并到main
git merge dev

# 5. 推送更新后的main
git push origin main
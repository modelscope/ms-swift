# 可以直接在终端里运行
for f in shard_*.tar.gz; do
  echo "正在解压 $f ..."
  tar -xzf "$f"
done

echo "全部解压完成。"
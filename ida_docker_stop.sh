# 批量删除所有匹配 ida_server_ 前缀的容器
docker ps -a --format "{{.Names}}" | grep "^ida_server_" | xargs -r docker rm -f
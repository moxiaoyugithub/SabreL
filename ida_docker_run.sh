#!/bin/bash

# ==============================================================================
# 脚本名称: ida_docker_run.sh
# 功能: 批量启动用于 IDA Server 的 Docker 容器
# 逻辑: 1. 解析模型列表 2. 检查容器是否存在 3. 存在则跳过，不存在则启动
# ==============================================================================

# 检查参数数量
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <SABRE_DIR> <ENV_NUM> <TARGET_TYPES>"
    echo "Example: $0 /data/mxy/GBO 8 \"[BinCola, CLAP, jTrans]\""
    exit 1
fi

SABRE_DIR=$1
ENV_NUM=$2

# 1. 处理 TARGET_TYPES：去掉中括号、空格、换行，并将逗号替换为空格以便 for 循环遍历
TARGET_TYPES_RAW=$3
TARGET_TYPES_CLEAN=$(echo $TARGET_TYPES_RAW | tr -d '[] ' | tr ',' ' ')

echo "--- 正在初始化分析环境 ---"
echo "项目目录: $SABRE_DIR"
echo "并行副本: $ENV_NUM (0 到 $((ENV_NUM-1)))"
echo "模型列表: $TARGET_TYPES_CLEAN"
echo "--------------------------"

# 2. 循环模型种类
for model in $TARGET_TYPES_CLEAN; do
    # 3. 循环环境编号
    for ((i=0; i<$ENV_NUM; i++)); do
        CONTAINER_NAME="ida_server_${model}_${i}"
        
        # 4. 检查容器是否已存在 (使用正则精确匹配全名)
        # -a 检查所有状态（Running, Exited, Created 等）
        if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
            # 获取容器当前状态
            STATUS=$(docker inspect -f '{{.State.Status}}' $CONTAINER_NAME)
            echo "[SKIP] 容器 $CONTAINER_NAME 已存在 (状态: $STATUS)，跳过..."
            
            # 可选：如果容器存在但处于 Exited 状态，你可能希望启动它
            if [ "$STATUS" == "exited" ]; then
                echo "       --> 检测到容器已停止，正在尝试启动..."
                docker start $CONTAINER_NAME > /dev/null
            fi
            continue
        fi

        # 5. 只有不存在时才执行 docker run
        echo "[RUN] 正在启动新容器: $CONTAINER_NAME ..."
        docker run -d \
            --name "$CONTAINER_NAME" \
            -v "$SABRE_DIR":/SABRE \
            -w /SABRE \
            ida-base:v2-with-license \
            tail -f /dev/null

        # 6. 检查启动结果
        if [ $? -eq 0 ]; then
            echo "[SUCCESS] $CONTAINER_NAME 启动成功"
        else
            echo "[FAILED] $CONTAINER_NAME 启动失败，请检查资源或 Docker 日志"
        fi
    done
done

echo "--------------------------"
echo "所有容器检查完毕。"
echo "提示: 如果看到大量跳过但 'docker ps' 没看到容器，请执行 'docker ps -a' 查看是否有 Exited 状态的残留。"
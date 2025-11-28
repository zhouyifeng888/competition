pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
cd /home/ma-user/work/task1
export no_proxy='a.test.com,127.0.0.1,2.2.2.2'
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install attrdict  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensordict  -i https://pypi.tuna.tsinghua.edu.cn/simple
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.6.0/MindSpore/unified/aarch64/mindspore-2.6.0-cp${PY_VER}-cp${PY_VER}-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://repo.huaweicloud.com/repository/pypi/simple
pip uninstall mindformers -y
if [ ! -d "mindnlp" ]; then
    echo "mindnlp目录不存在，正在克隆 mindnlp 仓库..."
    git clone https://gitee.com/mindspore-lab/mindnlp.git
else
    echo "mindnlp目录已存在，跳过克隆操作。"
fi
cd mindnlp
git checkout 0.4 # 创建容器后首次执行需要切换分支,容器重启则不需要
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

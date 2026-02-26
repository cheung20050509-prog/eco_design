#!/bin/bash
# ============================================================
# Optuna 超参数自动搜索脚本
# ============================================================
# 用法:
#   ./run_optuna.sh              # 默认: 100次搜索, 100只股票, 500 epochs
#   ./run_optuna.sh 50           # 50次搜索
#   ./run_optuna.sh 50 60        # 50次搜索, 60只股票
# ============================================================

set -e

# 参数解析
N_TRIALS=${1:-100}
N_STOCKS=${2:-100}

echo "========================================"
echo " Optuna 超参数自动搜索"
echo "========================================"
echo " 搜索次数: ${N_TRIALS}"
echo " 股票数量: ${N_STOCKS}"
echo " 训练轮数: 500 epochs (与main.py一致)"
echo "========================================"

# 激活conda环境（如果存在）
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate eco_design 2>/dev/null || true
fi

# 确保在项目目录
cd "$(dirname "$0")"

# 检查optuna是否安装
python -c "import optuna" 2>/dev/null || {
    echo "安装 optuna..."
    pip install optuna -q
}

# 设置PyTorch显存分配优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 运行搜索
echo "运行Optuna超参数搜索 (每个trial训练500 epochs)..."
python optuna_search.py \
    --trials ${N_TRIALS} \
    --stocks ${N_STOCKS}

echo ""
echo "========================================"
echo " 搜索完成！"
echo "========================================"
echo " 结果文件:"
echo "   output/optuna_best_params.json  - 最佳超参数"
echo "   output/optuna_all_trials.csv    - 所有trial记录"
echo "========================================"

#!/usr/bin/env bash
# 无人值守训练 + 诊断 + 出图 + 报告（tmux/nohup 启动）
# 用法: EXP_NAME=xxx TOTAL_EPOCHS=20 bash run_all.sh
set -u

EXP_NAME="${EXP_NAME:-overnight_$(date +%Y%m%d_%H%M%S)}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
VAL_METRICS_EVERY="${VAL_METRICS_EVERY:-3}"
MAX_RETRY="${MAX_RETRY:-3}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

mkdir -p exp_logs
LOG="exp_logs/train_$(date +%Y%m%d_%H%M%S).log"
EXP_DIR="experiments/$EXP_NAME"
EXC_FILE="$EXP_DIR/exceptions.log"

log()   { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
record_exc() { mkdir -p "$EXP_DIR"; echo "[$(date '+%H:%M:%S')] $*" >> "$EXC_FILE"; log "异常: $*"; }

latest_ckpt() {
    for f in "$EXP_DIR/models/latest.pth" "$EXP_DIR/models/best_model.pth"; do
        [ -f "$f" ] && { echo "$f"; return 0; }
    done
    echo ""
}

log "=============================================="
log "无人值守训练开始"
log "实验: $EXP_NAME | epoch=$TOTAL_EPOCHS | 每${VAL_METRICS_EVERY}epoch验证 | 重试上限=$MAX_RETRY"
log "日志: $LOG"
log "=============================================="

# ===== 训练（带自动恢复） =====
TRAIN_EXIT=1
RESUME_ARG=""
for attempt in $(seq 1 $((MAX_RETRY + 1))); do
    log "----- 训练尝试 $attempt/$((MAX_RETRY + 1)) -----"
    # shellcheck disable=SC2086
    python train.py --exp "$EXP_NAME" --total-epochs "$TOTAL_EPOCHS" \
        --val-metrics-every "$VAL_METRICS_EVERY" $RESUME_ARG >> "$LOG" 2>&1
    TRAIN_EXIT=$?
    if [ $TRAIN_EXIT -eq 0 ]; then
        log "训练正常退出（exit 0）"
        break
    fi
    record_exc "训练异常退出 exit=$TRAIN_EXIT"
    CKPT=$(latest_ckpt)
    if [ -n "$CKPT" ] && [ $attempt -le $MAX_RETRY ]; then
        log "用 checkpoint 恢复重试: $CKPT"
        RESUME_ARG="--resume $CKPT"
    else
        log "无可用 checkpoint 或已达最大重试，停止重试"
        break
    fi
done

# 记录 NaN/OOM 停止原因（train.py 会打印 [STOP]）
if grep -q "\[STOP\]" "$LOG" 2>/dev/null; then
    record_exc "$(grep '\[STOP\]' "$LOG" | tail -1)"
fi

# ===== 诊断 + 出图 + 报告（用最后有效 checkpoint） =====
CKPT=$(latest_ckpt)
export SAHI_MODEL_PATH="$CKPT"

log "=============================================="
log "训练结束（exit=$TRAIN_EXIT），开始诊断/出图/报告"
log "=============================================="

if [ -n "$CKPT" ]; then
    log "使用 checkpoint: $CKPT"

    python diagnose_bbox.py >> "$LOG" 2>&1 \
        && log "diagnose_bbox.py OK" || record_exc "diagnose_bbox.py 失败（继续）"

    python evaluate_and_visualize.py --n 10 >> "$LOG" 2>&1 \
        && log "evaluate_and_visualize.py OK" || record_exc "evaluate_and_visualize.py 失败（继续）"

    python make_report.py --exp "$EXP_NAME" --log "$LOG" >> "$LOG" 2>&1 \
        && log "make_report.py OK" || record_exc "make_report.py 失败（继续）"
else
    record_exc "无 checkpoint，跳过诊断/出图/报告"
fi

log "=============================================="
log "全部完成"
log "报告: experiments/$EXP_NAME/TRAIN_REPORT.md"
log "指标: experiments/$EXP_NAME/val_metrics.json"
log "=============================================="

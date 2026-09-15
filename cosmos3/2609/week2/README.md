# 2026-09 week2（09/09 – 09/15）

DROID Edge 整 episode teacher-forcing：加速战役 + 4 卡 token 上限。

| 文档 | 内容 |
| --- | --- |
| [weekly_summary.md](weekly_summary.md) | 周报。**§2 三个因素（DataLoader / VAE / 网络）**，**§3 TND 实现与利弊**；其后是代码、战役、默认、待办 |
| [tnd_q_kv.md](tnd_q_kv.md) | TND 的 Q / KV 怎么拼：`UND C1 C2 N1 N2`。Q 是 GEN 连续切片，KV 按可见段 gather |
| [comparison_tests.md](comparison_tests.md) | 开头同样是三因素 + TND；后面是全部对比测试的配置与结果表 |

上周：[../week1/weekly_summary.md](../week1/weekly_summary.md)（09/02–09/08，整 episode 跑通 + `max_episode_blocks` 扫参）。

原始 run 目录在仓库 `cosmos-framework/experiments/`。

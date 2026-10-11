"""
Global Dynamic Scene Feature Scoring Pool Classifier (场景打分池分类器).
Evaluates all registered scene candidates simultaneously,
applies confidence thresholding and difference-margin debouncing,
and selects the winning scene with highest confidence.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field

from .scene_types import SceneType
from .evaluators import SceneEvaluators

logger = logging.getLogger("ScenePool")

@dataclass
class SceneResult:
    scene: SceneType
    confidence: float
    second_scene: SceneType = SceneType.UNKNOWN
    second_score: float = 0.0
    margin: float = 0.0
    all_scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

class ScenePoolClassifier:
    def __init__(self, min_threshold: float = 50.0, min_margin: float = 15.0):
        self.min_threshold = min_threshold
        self.min_margin = min_margin
        self._last_stable_scene = SceneType.UNKNOWN
        self._last_stable_time = 0.0
        self._jitter_count = 0

    def reset_debounce(self) -> None:
        """
        清掉"上一帧稳定场景"的粘连（全局卡死守卫升级时调用）。

        为什么需要：防抖是为了滤掉噪声，但一旦它把一个**错的**稳定场景粘住，
        后面每一轮都会优先沿用它 -> 越卡越死。守卫升级到级别 1 时强制松手，让下一轮重新判。
        """
        self._last_stable_scene = SceneType.UNKNOWN
        self._jitter_count = 0
        self._last_stable_time = 0.0

    def classify(self, img_bgr: np.ndarray, text_corpus: str, ocr_targets: list) -> SceneResult:
        """
        Run all 9 multi-modal scene evaluators across the pool and return the winner.
        """
        if img_bgr is None or img_bgr.size == 0:
            return SceneResult(scene=SceneType.UNKNOWN, confidence=0.0)

        # 1. Run all candidate evaluators
        candidates = [
            (SceneType.ROGUELIKE_MAP, SceneEvaluators.eval_roguelike_map),
            (SceneType.EVENT_CHOICE, SceneEvaluators.eval_event_choice),
            (SceneType.EPIPHANY_HERO, SceneEvaluators.eval_epiphany_hero),
            (SceneType.EPIPHANY_CARDS, SceneEvaluators.eval_epiphany_cards),
            (SceneType.BATTLE, SceneEvaluators.eval_battle),
            (SceneType.CAMP_SHOP, SceneEvaluators.eval_camp_shop),
            (SceneType.EQUIP_ASSIGN, SceneEvaluators.eval_equip_assign),
            (SceneType.STAGE_MAP, SceneEvaluators.eval_stage_map),
            (SceneType.CUTSCENE_DIALOG, SceneEvaluators.eval_cutscene_dialog),
            (SceneType.SETTLEMENT_LOOT, SceneEvaluators.eval_settlement_loot),
            (SceneType.SYSTEM_ZERO_LOBBY, SceneEvaluators.eval_system_zero_lobby),
            (SceneType.CODEX_SELECT, SceneEvaluators.eval_codex_select),
            (SceneType.HOME_LOBBY, SceneEvaluators.eval_home_lobby),
            (SceneType.CHAOS_SELECT, SceneEvaluators.eval_chaos_select),
            (SceneType.DREAM_MODE, SceneEvaluators.eval_dream_mode),
            (SceneType.TREATMENT_CENTER, SceneEvaluators.eval_treatment_center),
        ]

        scored: List[Tuple[SceneType, float, Dict[str, Any]]] = []
        scores_map: Dict[str, float] = {}

        for scene_type, eval_fn in candidates:
            score, det = eval_fn(img_bgr, text_corpus, ocr_targets)
            scored.append((scene_type, score, det))
            scores_map[scene_type.value] = round(score, 1)

        return self._decide(scored, scores_map)

    def _decide(self, scored: List[Tuple[SceneType, float, Dict[str, Any]]],
                scores_map: Dict[str, float]) -> SceneResult:
        """把"各候选分数"变成单一结论（含阈值、防抖、兜底）。抽出来是为了能单测。"""
        scored = sorted(scored, key=lambda x: x[1], reverse=True)
        best_scene, best_score, best_det = scored[0]
        second_scene, second_score, _ = scored[1] if len(scored) > 1 else (SceneType.UNKNOWN, 0.0, {})
        margin = best_score - second_score

        # Log Top 2 candidates for debugging
        if best_score > 30.0:
            logger.debug(f"【打分池竞争】第一: [{best_scene.value}] ({best_score:.1f}分) | 第二: [{second_scene.value}] ({second_score:.1f}分) | 分差: {margin:.1f}")

        # 2. Threshold & Debounce Logic
        # Condition A: Winner is confident (>= min_threshold) and leads by >= min_margin
        if best_score >= self.min_threshold and margin >= self.min_margin:
            self._last_stable_scene = best_scene
            self._last_stable_time = time.time()
            self._jitter_count = 0
            return SceneResult(
                scene=best_scene,
                confidence=best_score,
                second_scene=second_scene,
                second_score=second_score,
                margin=margin,
                all_scores=scores_map,
                details=best_det
            )

        # Condition B: High absolute score (>= 80.0) wins directly
        if best_score >= 80.0:
            self._last_stable_scene = best_scene
            self._last_stable_time = time.time()
            self._jitter_count = 0
            return SceneResult(
                scene=best_scene,
                confidence=best_score,
                second_scene=second_scene,
                second_score=second_score,
                margin=margin,
                all_scores=scores_map,
                details=best_det
            )

        # Condition C: Ambiguous / Low score transition (防抖缓冲: 最多保持上一稳定状态 2 帧)
        #
        # 重要修正（真机教训）：**只有"新结论本身不自信"时才允许防抖**。
        # 那轮日志：屏幕上已经是角色对白（剧情对白演出 75 分），只因第二候选 65 分、分差 10 < 15，
        # 防抖就把上一帧的 [外层选关] 强行保留了 2 帧 → 脚本去"拖动地图" → 20 秒后看门狗 ESC。
        # 分差小只说明"有两个都不错的候选"，此时跟随最强候选，比抱住过期场景正确得多。
        if best_score < self.min_threshold and self._last_stable_scene != SceneType.UNKNOWN and self._jitter_count < 2:
            self._jitter_count += 1
            logger.info(f"【打分防抖】首位分数不足 ({best_score:.1f} < {self.min_threshold})，"
                        f"维持上一帧稳定场景: [{self._last_stable_scene.value}] (第 {self._jitter_count} 帧缓冲)")
            return SceneResult(
                scene=self._last_stable_scene,
                confidence=best_score,
                second_scene=second_scene,
                second_score=second_score,
                margin=margin,
                all_scores=scores_map,
                details={"debounced": True}
            )

        # 分数够但分差不足 -> 直接采用最强候选（不再抱住过期场景）
        if best_score >= self.min_threshold:
            self._last_stable_scene = best_scene
            self._last_stable_time = time.time()
            self._jitter_count = 0
            return SceneResult(
                scene=best_scene,
                confidence=best_score,
                second_scene=second_scene,
                second_score=second_score,
                margin=margin,
                all_scores=scores_map,
                details={**best_det, "low_margin": round(margin, 1)}
            )

        # Fallback to UNKNOWN
        self._last_stable_scene = SceneType.UNKNOWN
        return SceneResult(
            scene=SceneType.UNKNOWN,
            confidence=best_score,
            second_scene=second_scene,
            second_score=second_score,
            margin=margin,
            all_scores=scores_map,
            details={"reason": "low_confidence_or_tie"}
        )

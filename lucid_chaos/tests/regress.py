"""
离线回归夹具：把已有截图当作"冻结的行为快照"，每次重构后跑一遍，
确认「路由到哪个场景 + 第一个动作是什么」没有意外变化。

用法（在工作区根目录）：
    python -m lucid_chaos.tests.regress            # 跑全部
    python -m lucid_chaos.tests.regress rogue      # 只跑名字含 rogue 的

约定：
  - 这里**不发任何真实点击**，输入层全部打桩。
  - `status='ok'`          = 当前行为已人工确认正确，重构不许改坏
  - `status='known_bad'`   = 已知行为不对但先冻结（重构后若变化，diff 会明确提示，便于有意修正）
"""
import sys
import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import cv2
from lucid_chaos.core.vision_service import VisionService
from lucid_chaos.state.observer import ScreenObserver
from lucid_chaos.router.router import SceneRouter
from lucid_chaos.logic.node_engine import NodeDecisionEngine
from lucid_chaos.logic.path_planner import PathPlanner
from lucid_chaos.logic.story_pusher import StoryPusher

FIXTURES = _HERE.parent / "fixtures"

# name, file, 期望路由, 期望动作标签(任一命中即可), status, 说明
CASES = [
    ("narration_with_markers", "narration_with_markers.png", "局内互动事件", "【探索】交互点 1 (F1)", "known_bad",
     "旁白+搜索标记：现在按 F1 触发交互点（主人定的规则：探索 = F1/F2/F3），不再点世界坐标"),
    ("event_box_end_search", "event_box_end_search.png", "局内互动事件", "【事件框】选项 1", "ok",
     "底部「結束搜查/觸發事件」确认框"),
    ("event_box_survival_signal", "event_box_survival_signal.png", "局内互动事件", "【事件框】选项 1", "ok",
     "底部「尋找生存訊號」事件框"),
    ("card_detail_popup", "card_detail_popup.png", "局内互动事件", "【弹窗】", "ok",
     "查看詳情弹窗 -> 点底部中央橙色【關閉】"),
    ("card_reward_3choose", "card_reward_3choose.png", "灵光一闪选牌", "【选牌】确认(ENTER)", "ok",
     "卡牌奖励 3选1 + 確認"),
    ("epiphany_hero_select", "epiphany_hero_select.png", "灵光一闪选人", "【灵光一闪】确认战斗员(ENTER)", "ok",
     "选择欲获得灵光一闪的战斗员界面 -> 智能轮换 1/2/3 号位 + ENTER 确认"),
    ("epiphany_god_flash", "epiphany_god_flash.png", "灵光一闪选牌", "【选牌·神闪】", "ok",
     "灵光一闪神闪（金色包边卡牌） -> 最高优先级识别并选定神闪卡牌"),
    ("card_reward_assign", "card_reward_assign.png", "获得装备分配", "【中立卡跳过】", "ok",
     "卡牌奖励分配 -> 按主人规则跳过中立卡(F1)，不存入角色"),
    ("card_reward_hero_assign", "card_reward_hero_assign.png", "获得装备分配", "【卡牌领奖】选择第1战斗员", "ok",
     "卡牌奖励分配（角色可用卡） -> 准确选择战斗员并按 ENTER 确认交付（防 [移除] 标签误判为删牌）"),
    ("rogue_map_bright", "rogue_map_bright.png", "肉鸽选路", "【肉鸽选路】第 1 条路线", "ok",
     "肉鸽选路（亮）"),
    ("rogue_map_dim", "rogue_map_dim.png", "肉鸽选路", "【肉鸽选路】第 1 条路线", "ok",
     "肉鸽选路（压暗过渡帧，靠青色面板占比兜底）"),
    ("stage_map_chapter1", "stage_map_chapter1.png", "外层选关", "【外层选关】", "ok",
     "外侧选关（雪白大地图 + 一排关卡徽章；曾被'剧情对白演出'抢走路由去点屏幕中央）"),
    ("stage_map_drawer", "stage_map_drawer.png", "外层选关", "【关卡详情】", "ok",
     "选关界面点开详情抽屉（真机自动取证；应点右下【進入】）"),
    ("team_config_enter", "team_config_enter.png", "外层选关", "【配置隊伍】進入", "ok",
     "配队界面（文字锁定 L1：OCR 读到 `配置隊伍` 直接定案）：曾被'局内互动事件(65)'抢走路由 -> 去点别的地方"),
    ("stage_map_drawer_text_trap", "stage_map_drawer_text_trap.png", "外层选关", "【关卡详情】進入", "ok",
     "关卡详情抽屉·**关键字陷阱**：正文『和阿黛海特一起進入夢中的世界』含「進入」-> 旧写法 locate_text 命中正文行"
     "(0.742,0.335)、位置校验失败 -> 反复点节点、永远进不去。现改为**区域限定** + 右下宽按钮形状兜底"),
    ("settlement_stage_reward", "settlement_stage_reward.png", "战后结算战利品", "【关卡结算】", "ok",
     "关卡奖励结算（紫色【確認】+ 右上橙色 XP 条；曾被'局内互动事件'抢走路由 -> 点屏幕中央卡死）"),
    ("settlement_teal_hero", "settlement_teal_hero.png", "战后结算战利品", "【关卡结算】", "ok",
     "关卡奖励结算·青发角色版：曾被'肉鸽选路'(100) 抢走路由 —— 角色青色头发/光环骗过了青色判据，"
     "现改用**暗青**(V<150) + 右上橙色 XP 条反否决"),
    ("dialogue_lei", "dialogue_lei.png", "剧情对白演出", "【常驻加速】", "ok",
     "角色对白（真机自动取证）：handler 先点右上角 2X 加速，再推进台词"),
    ("dialogue_black_bg", "dialogue_black_bg.png", "剧情对白演出", "【主线剧情】", "ok",
     "**黑底白字剧情演出**（真机自动取证）：均值 6.4 与'全黑过渡帧'(3.3) 几乎一样，"
     "曾被当成过渡帧 -> 什么都不做 -> 看门狗硬停止。现判据=近黑 **且 有游戏文字**（排除脚本角标）"),
    ("rogue_map_real", "rogue_map_real.png", "肉鸽选路", "【肉鸽选路】第 2 条路线", "ok",
     "真机肉鸽选路（带对白叠加；模板失配时靠青色面板占比 3.52% 兜底）"),
    ("transition_black", "transition_black.png", "__ANY__", "__NONE__", "ok",
     "淡入淡出全黑过渡帧（真机自动取证）：**必须不动作**，否则会拿上一帧场景在黑屏上乱点"),
    ("event_phantom_choice", "event_phantom_choice.png", "局内互动事件", "【事件框】", "ok",
     "局内双选项事件（跟隨中央/右側幻影）：点第 1 个框；右上 ⏩ 未点亮时另由常驻加速补刀"),
    ("common_interact_mushroom", "common_interact_mushroom.png", "局内互动事件", "【事件框】", "ok",
     "局内交互场景（刺激蘑菇/觸發【費洛蒙孢子】事件）：**蘑菇森林的青色环境光**让'肉鸽选路'拿到 65 分"
     "并与局内事件打平 -> 被路径规划器抢走 -> 去点空地。现加'交互提示框字样'反否决 + 补齐事件关键字"),
    ("rogue_proceed_door", "rogue_proceed_door.png", "__ANY__", "【肉鸽前进】繼續進行(ENTER)", "ok",     "肉鸽里的'继续前进'门（真机截图）：右下角 `ENTER 繼續進行`（OCR 把 繼續 读成 缆）。"
     "曾被判成肉鸽选路(青色洞穴地面让 cyan 到 27.5%)，路径规划器在**没有任何分支**时盲按了 2 号键。"
     "现规则：**右下角出现 ENTER 按键提示 -> 直接按 ENTER**（游戏自己标的快捷键），"
     "并且'分支全 EMPTY 时绝不盲按'"),
    ("battle_hand_row", "battle_hand_row.png", "战斗中", "【战斗】", "ok",
     "战斗界面（真机截图）：曾被'局内互动事件(65)'抢走路由 -> 把手牌当事件框逐个点。"
     "现用形状判据：底部手牌边缘密度 12.04%（其它夹具 <=4.93%）"),
    ("battle_ep_insufficient", "battle_ep_insufficient.png", "战斗中", "【战斗】取消選擇(ESC)", "ok",
     "**EP 不够时按了卡/技能**弹出的「自我意識點數不足」选择界面（真机截图）："
     "左侧 ENTER 使用卡牌 / ESC 取消選擇。脚本会卡在这个选择态 -> 现统一按 **ESC** 退出，"
     "并且出牌前先读 EP、买不起就不按"),
    ("battle_endturn_confirm", "battle_endturn_confirm.png", "战斗中", "【战斗】確認结束回合(ENTER)", "ok",
     "**按 E 结束回合时**弹出的确认框「目前還有可用的卡牌在手牌中 / 是否要在不使用它們的情況下結束這回合?」"
     "（左 ESC 取消 / 右 ENTER 確認，真机截图）-> 我们已决定结束回合，直接按 **ENTER** 確認"),
    ("event_kaleidoscope_desire", "event_kaleidoscope_desire.png", "局内互动事件", "【事件框】选项 1", "ok",
     "肉鸽里的探索事件框「注視希爾德的欲望 / 隨機獲得1張欲望卡牌」+ 右下角通用 `⚔ ENTER` 图标。"
     "曾有两处错：① 我加的『右下角有 ENTER 字样就按 ENTER』规则太宽 -> 抢掉真正动作，已删除，"
     "改成只认具体文案『繼續進行』；② 观察者判 VICTORY_SETTLEMENT(95) 压过路由(65) -> 已改成"
     "『谁分高听谁的』。正确动作 = 按 **1** 选交互点"),
]


class RecordingInput:
    """记录被调用的点击，不做任何真实输入。"""
    def __init__(self):
        self.calls = []

    def click_client(self, cx, cy, **kw):
        self.calls.append(("click_client", int(cx), int(cy), kw.get("label")))

    def click_rel(self, rx, ry, **kw):
        self.calls.append(("click_rel", round(float(rx), 3), round(float(ry), 3), kw.get("label")))

    def click_area_rel(self, a, b, c, d, **kw):
        self.calls.append(("click_area_rel", round((a + c) / 2, 3), round((b + d) / 2, 3), kw.get("label")))

    def click_box(self, bbox, **kw):
        self.calls.append(("click_box", bbox, None, kw.get("label")))

    def click_target(self, target, **kw):
        c = getattr(target, "center", None)
        self.calls.append(("click_target", None if c is None else int(c[0]),
                           None if c is None else int(c[1]), kw.get("label")))

    def drag_rel(self, s_rx, s_ry, e_rx, e_ry, **kw):
        self.calls.append(("drag_rel", round(float(s_rx), 3), round(float(s_ry), 3), kw.get("label")))

    def drag_client(self, sx, sy, ex, ey, **kw):
        self.calls.append(("drag_client", int(sx), int(sy), kw.get("label")))

    # ---- 键盘路径（主人确认"键盘更准"后新增）----
    def tap_action(self, action, **kw):
        from lucid_chaos.core.keys import ACTION_KEYS
        key = ACTION_KEYS.get(action, "?")
        self.calls.append(("tap_action", action, key, kw.get("label") or f"【按键】{action}({key})"))
        return True

    def tap_card(self, index, **kw):
        self.calls.append(("tap_card", int(index), None, kw.get("label") or f"【按键】出牌 {index}"))
        return True

    def tap_hero_skill(self, slot, **kw):
        self.calls.append(("tap_hero_skill", int(slot), None, kw.get("label") or f"【按键】{slot}号角色技能"))
        return True

    def send_key(self, vk, **kw):
        self.calls.append(("send_key", int(vk), None, kw.get("label")))

    def __getattr__(self, item):
        # 兜底：未来新增的点击方法不会让夹具崩掉，同时把名字记下来便于发现"夹具盲区"
        if item.startswith("_"):
            raise AttributeError(item)
        def _rec(*a, **k):
            self.calls.append((f"other:{item}", None, None, k.get("label")))
        return _rec

    def send_key(self, vk, **kw):
        self.calls.append(("send_key", vk, None, kw.get("label")))

    def drag_rel(self, *a, **kw):
        self.calls.append(("drag_rel", None, None, None))


class StubCapture:
    """桩截屏：永远返回当前夹具图（生产里 handler 二次截屏拿到的是同一个画面）。"""
    def __init__(self, img):
        self.img = img

    def capture(self):
        return self.img


class StubCtx:
    def __init__(self, vision, img=None):
        self.vision = vision
        self.capture = StubCapture(img)
        self.overlay = None
        self.input = RecordingInput()


def imread_u(p: Path):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def main(argv):
    filt = argv[1] if len(argv) > 1 else ""
    vision = VisionService()
    observer = ScreenObserver(vision)
    path_planner = PathPlanner(preference="non_battle")

    rows, failed = [], 0
    for name, fname, exp_route, exp_label, status, note in CASES:
        if filt and filt not in name:
            continue
        path = FIXTURES / fname
        if not path.exists():
            rows.append((name, "MISSING", "-", "-", "-", f"夹具不存在: {fname}"))
            failed += 1
            continue

        img = imread_u(path)
        ctx = StubCtx(vision, img)
        test_cfg = {"main_story": {"boss_action": "auto"}, "card_reward": {"skip_neutral_cards": True}}
        engine = NodeDecisionEngine(ctx, None, test_cfg)
        engine._evt_sig = None
        engine._evt_stuck = 0
        # 用真实的装配方式：StoryPusher 自带 router，并把它自己的 node_engine 换成本地实例
        # （否则 STAGE_MAP / CUTSCENE / SETTLEMENT 这些分支因为没有 story_pusher 而不会执行任何动作）
        pusher = StoryPusher(ctx, test_cfg)
        pusher.node_engine = engine
        pusher.path_planner = path_planner
        router = pusher.router
        router.node_engine = engine
        router.path_planner = path_planner
        # 清掉可能跨夹具残留的计数器
        pusher.map_swipe_count = 0

        state = observer.observe(img)
        router.dispatch(img, state)

        # 路由：用**实际使用的**场景（dispatch 会写进 state.route，含 L1 文字锁定）
        # 以前这里直接问打分池，于是"文字锁定"这类定案在夹具里看不见（会误报失败）
        actual_route = getattr(state, "route", None)
        if actual_route is not None:
            got_route = actual_route.value
        else:
            res = router.classifier.classify(img, " ".join(state.raw_texts), state.targets)
            got_route = res.scene.value
        labels = [c[3] or "" for c in ctx.input.calls]
        got_label = next((l for l in labels if l), "(无动作)")
        first = ctx.input.calls[0] if ctx.input.calls else None
        coord = ""
        if first is not None:
            coord = f" @({first[1]},{first[2]})" if isinstance(first[1], int) else f" @rel({first[1]},{first[2]})"

        route_ok = (exp_route == "__ANY__") or (got_route == exp_route)
        if exp_label == "__NONE__":
            label_ok = len(ctx.input.calls) == 0          # 期望：什么都不点（过渡帧）
        else:
            label_ok = any(exp_label in l for l in labels)
        ok = route_ok and label_ok
        if not ok:
            failed += 1
        rows.append((name, "PASS" if ok else "FAIL",
                     got_route + ("" if route_ok else f" (期望 {exp_route})"),
                     got_label + coord + ("" if label_ok else
                                          (" (期望：不动作)" if exp_label == "__NONE__" else f" (期望含 {exp_label})")),
                     status, note))
    print("\n=== 离线回归夹具 ===")
    for name, verdict, route, label, status, note in rows:
        print(f"[{verdict}] {name}  <{status}>")
        print(f"        路由: {route}")
        print(f"        动作: {label}")
        if verdict == "FAIL":
            print(f"        期望: 路由={dict((c[0], c[2]) for c in CASES).get(name)}  动作含={dict((c[0], c[3]) for c in CASES).get(name)}")
        print(f"        说明: {note}")
    print(f"\n共 {len(rows)} 例，失败 {failed} 例")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

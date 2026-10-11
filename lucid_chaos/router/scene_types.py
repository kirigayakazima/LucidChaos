"""
Scene Taxonomy & Types for Global Dynamic Routing.
"""

from enum import Enum

class SceneType(Enum):
    STAGE_MAP = "外层选关"          # 外层大地图选关（章节/关卡抽屉）
    ROGUELIKE_MAP = "肉鸽选路"      # 内部肉鸽选路（3D地块 + 左下小地图）
    EVENT_CHOICE = "局内互动事件"   # 局内互动与事件（对话气泡选项、放大镜搜索）
    EPIPHANY_HERO = "灵光一闪选人"   # 灵光一闪选战斗员（请选择欲获得【灵光一闪】的战斗员）
    EPIPHANY_CARDS = "灵光一闪选牌" # 局内灵光一闪（真实3选1卡牌效果挑选）
    BATTLE = "战斗中"              # 战斗中（AUTO/技能/回合出牌）
    CAMP_SHOP = "营地与商店"        # 营地与商店（篝火免费休息/德朗商店安全区）
    EQUIP_ASSIGN = "获得装备分配"   # 获得装备分配（战斗员穿装）
    CUTSCENE_DIALOG = "剧情对白演出" # 剧情对白与演出（右上▷|跳过/台词推进）
    SETTLEMENT_LOOT = "战后结算战利品" # 战后结算与战利品（獲得/確認按钮）
    HOME_LOBBY = "游戏大厅主页"        # 主界面大厅（出击/模拟/卡厄思/方舟都市/战斗员）
    CHAOS_SELECT = "卡厄思分支选择"    # 卡厄思页面（卡厄思显现 vs 零号系统）
    SYSTEM_ZERO_LOBBY = "零号系统主界面" # 零号系统主界面（备援协议/升级/法典入口）
    CODEX_SELECT = "零号法典选择"     # 零号系统法典挑选（01/02/03号法典 + 進入）
    DREAM_MODE = "夢境模式"            # 梦境模式（深层创伤启动梦境，点击左下治疗中心）
    TREATMENT_CENTER = "治療中心"      # 治疗中心（选择受创战斗员并消除记忆）
    UNKNOWN = "未知巡航"            # 未知过渡画面


# -------------------------------------------------------------
# 两套场景分类的映射：router(SceneType) <-> observer(ScreenState)
# 背景：改造前 observer 与 router 各判各的、可以互相矛盾
# （实测卡牌奖励图：observer=EPIPHANY_SELECTION，路由=未知巡航75）。
# 这里先把两者的对应关系写死，再由 SceneRouter 按 config.logging/classification.authority
# 决定「谁说了算」；默认仍是 observer 说了算（行为不变），等真机验证后再切换。
# -------------------------------------------------------------
from ..state.screen_state import ScreenState  # noqa: E402  (放在枚举定义之后，避免循环导入)

SCENE_TO_SCREEN = {
    SceneType.STAGE_MAP: ScreenState.MAP_NAVIGATION,
    SceneType.ROGUELIKE_MAP: ScreenState.MAP_NAVIGATION,
    SceneType.EVENT_CHOICE: ScreenState.EVENT,
    SceneType.EPIPHANY_HERO: ScreenState.EPIPHANY_SELECTION,
    SceneType.EPIPHANY_CARDS: ScreenState.EPIPHANY_SELECTION,
    SceneType.BATTLE: ScreenState.BATTLE,
    SceneType.CAMP_SHOP: ScreenState.CAMP,
    SceneType.EQUIP_ASSIGN: ScreenState.EQUIP_SELECTION,
    SceneType.CUTSCENE_DIALOG: ScreenState.DIALOG,
    SceneType.SETTLEMENT_LOOT: ScreenState.VICTORY_SETTLEMENT,
    SceneType.HOME_LOBBY: ScreenState.MAP_NAVIGATION,
    SceneType.CHAOS_SELECT: ScreenState.MAP_NAVIGATION,
    SceneType.SYSTEM_ZERO_LOBBY: ScreenState.MAP_NAVIGATION,
    SceneType.CODEX_SELECT: ScreenState.MAP_NAVIGATION,
    SceneType.DREAM_MODE: ScreenState.MAP_NAVIGATION,
    SceneType.TREATMENT_CENTER: ScreenState.MAP_NAVIGATION,
    SceneType.UNKNOWN: ScreenState.UNKNOWN,
}

SCREEN_TO_SCENE = {v: k for k, v in SCENE_TO_SCREEN.items()}
# 一个 ScreenState 可能对应多个 SceneType（如 MAP_NAVIGATION 同时对应外层选关与肉鸽选路），
# 反查只用于诊断，取第一个即可。


# -------------------------------------------------------------
# 「局内常驻 2X 加速」补刀动作适用哪些场景（主人 2026/10/6 要求："局内应该帮我激活"）
# 明确排除：
#   - CUTSCENE_DIALOG：对白界面由 handle_cutscene_dialog 自己点（再点一次会把开关关掉）
#   - BATTLE：AUTO 由 BattleManager 管（同上，重复点会切回去）
#   - STAGE_MAP / UNKNOWN：外层选关与未知画面不该碰顶部图标
# -------------------------------------------------------------
SPEED_HOOK_SCENES = {
    SceneType.EVENT_CHOICE,
    SceneType.EPIPHANY_HERO,
    SceneType.EPIPHANY_CARDS,
    SceneType.EQUIP_ASSIGN,
    SceneType.ROGUELIKE_MAP,
    SceneType.SETTLEMENT_LOOT,
    SceneType.CAMP_SHOP,
}


def should_activate_speed(scene) -> bool:
    """该场景是否应交由"局内常驻 2X"这道补刀动作处理。"""
    return scene in SPEED_HOOK_SCENES


# -------------------------------------------------------------
# L1「文字锁定」表：稳定 UI 文字直接定案，**不参加打分池竞争**
#
# 为什么需要它（真机教训）：配队界面 OCR 明明读到了 `配置伍（配置隊伍）/ 登場怪物 / 等級x3 / 進入`，
# 但它去参加了打分池：`局内互动事件 65`（和肉鸽选路 65 打平）赢出 -> 脚本去点别的地方，
# 而代码里本来就有正确的 `【配置隊伍】` 分支（点右下【進入】(0.935,0.913)）却没被调用。
# 结论：**这类界面文字极其可靠，就该用文字限定死，而不是让形状/颜色判据去猜。**
#
# 规则形式：`((必须同时命中的关键字组...), 场景, 说明)`
#   - 组内任一词命中即可；组间是 AND（组合条件，避免单个通用词误伤）
#   - 匹配是**模糊匹配**（实测 `配置隊伍` 被 OCR 读成 `配置伍`，精确子串会漏）
#   - 顺序敏感：更具体的规则写在前面
# -------------------------------------------------------------
TEXT_LOCK_RULES = (
    # 梦境模式（深层创伤启动）：`夢境模式` + (`治療中心` / `精神異常` / `崩潰加劇` / `靈光一閃`)，或 `深層創傷` + `治療中心`
    ((("夢境模式", "梦境模式", "啟動夢境模式", "启动梦境模式"),
      ("治療中心", "治疗中心", "精神異常", "精神异常", "崩潰加劇", "崩溃加剧", "靈光一閃", "灵光一闪", "精神翼常")), SceneType.DREAM_MODE,
     "梦境模式 -> 点击左下【治療中心】进行治疗"),
    ((("深層創傷", "深层创伤", "深创傷", "深唇創傷"), ("治療中心", "治疗中心")), SceneType.DREAM_MODE,
     "梦境模式 -> 点击左下【治療中心】进行治疗"),
    # 治疗中心（消除记忆）：`治療中心` 伴随 `消除記憶`/`諮商`/`深層創傷`，或消除确认弹窗（双重约束防误锁）
    ((("治療中心", "治疗中心"), ("消除記憶", "消除记忆", "諮商", "咨询", "消除", "記憶", "记忆", "深層創傷", "深层创伤", "戰鬥員", "战斗员")), SceneType.TREATMENT_CENTER,
     "治疗中心 -> 选择创伤角色并点击【消除記憶】"),
    ((("消除記憶", "消除记忆"), ("諮商", "咨询", "深層創傷", "深层创伤", "取消", "費用")), SceneType.TREATMENT_CENTER,
     "治疗中心 -> 选择创伤角色并点击【消除記憶】"),
    ((("損壞資料", "损坏资料", "即將消滅", "即将消灭", "無法復原", "无法复原"),), SceneType.TREATMENT_CENTER,
     "治疗中心 -> 弹窗消除記憶二次确认"),
    # 配队界面：`配置隊伍`(模糊，防漏字配置伍) + `進入`(按钮) 同时出现 -> 就是配队界面
    ((("配置隊伍", "配置队伍", "配置伍", "登場怪物", "登场怪物"), ("進入", "进入", "免費", "免费")), SceneType.STAGE_MAP,
     "配队界面 -> 点右下【進入】"),
    # 关卡详情抽屉：`關卡資訊` + `進入`
    ((("關卡資訊", "关卡资讯"), ("進入", "进入")), SceneType.STAGE_MAP,
     "关卡详情抽屉 -> 点右下【進入】"),
    # 命运选择界面：`選擇您的命運` / `您的命運` / `選擇命運` / `欲適用的命運` (3选1卡牌，禁绝单字"命運"，避免误伤法典描述)
    ((("選擇您的命運", "选择您的命运", "您的命運", "您的命运", "選擇命運", "选择命运", "欲適用的命運", "欲适用的命运"),), SceneType.EPIPHANY_CARDS,
     "命运选择界面 -> 3选1选牌"),
    # 灵光一闪选择战斗员：`請選擇欲獲得【靈光一閃】的戰鬥員` / (`靈光一閃` + `欲獲得` / `戰鬥員` / OCR 漂移 `撐欲瘦得` + `的門眞`)
    ((("請選擇欲獲得【靈光一閃】的戰鬥員", "請選擇欲獲得靈光一閃的戰鬥員", "欲獲得【靈光一閃】的戰鬥員", "欲獲得靈光一閃的戰鬥員", "欲獲得【靈光一閃】", "欲獲得靈光一閃", "欲獲得", "欲获得", "撐欲瘦得", "撑欲瘦得"),
      ("靈光一閃", "灵光一闪", "【靈光一閃】", "【灵光一闪】", "靈光", "灵光", "【靈光一】", "靈光一"),
      ("戰鬥員", "战斗员", "戰鬥", "战斗", "隊員", "队员", "確認", "确认", "ENTER", "門眞", "门真", "的門眞", "的门真", "的眞", "的真")), SceneType.EPIPHANY_HERO,
     "灵光一闪选人 -> 选择欲获得灵光一闪的战斗员并确认"),
    # 触发灵光一闪卡牌选择：`請選擇1張欲觸發靈光一閃的卡牌` / `欲觸發靈光一閃` / (`選擇卡牌` + `SPACE` + `靈光一閃`)
    ((("欲觸發靈光一閃", "欲触发灵光一闪", "觸發靈光一閃", "触发灵光一闪", "欲發靈光", "欲发灵光", "欲觸發", "欲触发", "選擇卡牌", "选择卡牌"),
      ("靈光一閃", "灵光一闪", "靈光", "灵光")), SceneType.EPIPHANY_CARDS,
     "触发灵光一闪选牌 -> 从卡组选择卡牌触发"),
    # 灵光一闪选牌（3选1效果 / 靈光一閃：角色名 / CAPS 查看內容）
    ((("靈光一閃", "灵光一闪", "靈光一", "灵光一", "靈光", "灵光"),
      ("查看內容", "查看内容", "CAPS", "效果選擇", "效果选择")), SceneType.EPIPHANY_CARDS,
     "灵光一闪选牌 -> 3选1效果挑选"),
    # 零号系统法典选择：`零號系統` + `法典` + `進入`
    ((("零號系統", "零号系统", "零号系統", "零號系统"), ("法典", "撑法典", "選擇法典", "选择法典", "選撑法典"), ("進入", "进入")), SceneType.CODEX_SELECT,
     "零号系统法典选择 -> 点选目标法典并进入"),
    # 零号系统大厅：`零號系統` + `法典` + (`備援協議` / `法典紀錄` / `升級`)
    ((("零號系統", "零号系统", "零号系統", "零號系统"), ("法典",), ("備援協議", "备援协议", "備援協諾", "法典紀錄", "法典纪录", "法典紀", "升級", "升级")), SceneType.SYSTEM_ZERO_LOBBY,
     "零号系统大厅 -> 点击法典图标"),
    # 获得装备分配：`裝備` + (`請選擇裝備` / `選擇裝備` / `請選擇戰鬥員` / `選擇戰鬥員` / `提煉` / `0/120pt`)
    ((("裝備", "装备", "装備"), ("請選擇裝備", "请选择装备", "選擇裝備", "选择装备", "请摆装備", "請擺裝備", "選擇戰鬥員", "选择战斗员", "請選擇戰鬥員", "请选择战斗员", "提煉", "提炼", "提烦", "0/120pt")), SceneType.EQUIP_ASSIGN,
     "获得装备分配 -> 选择装备或分配给战斗员"),
    # 卡牌奖励战斗员分配：`卡牌` + `欲領取卡牌` + `戰鬥員`（三组强约束，防 3选1 选牌误伤）
    ((("卡牌", "卡牌樊", "卡牌奖", "卡牌獎勵", "卡牌奖励"),
      ("欲領取卡牌", "欲领取卡牌", "領取卡牌", "领取卡牌", "撑欲取卡牌", "撐欲取卡牌", "欲領取", "欲领取"),
      ("戰鬥員", "战斗员", "戰斗員", "載", "载", "戟国眞")), SceneType.EQUIP_ASSIGN,
     "卡牌奖励分配 -> 选择有效战斗员收下卡牌并确认"),
    # 装备重铸 / 神之锤：必须命中标题/专属核心词(`重鑄裝備`/`神之錘`/`使用神之`) + 系统控制项(`120pt`/`重新整理`/`決定`/`欲套用`/`強化的裝備`)
    ((("重鑄裝備", "重装备", "重装備", "神之錘", "神之锤", "神之鈺", "神之槌", "使用神之"),
      ("120pt", "pt", "重新整理", "決定", "决定", "欲套用", "強化的裝備", "强化的装备")), SceneType.EQUIP_ASSIGN,
     "装备重铸/神之锤 -> 选择重铸效果或可强化装备并确认"),
    # 卡厄思分支选择：`卡厄思顯現` + `零號系統`
    ((("卡厄思顯現", "卡厄思显现"), ("零號系統", "零号系统", "零号系統", "零號系统")), SceneType.CHAOS_SELECT,
     "卡厄思分支选择 -> 点击零号系统"),
    # 游戏主界面大厅：必须同时命中大厅专属右侧栏(`方舟都市`/`救援`) + 核心模式(`卡厄思`/`出擊`/`模擬`/`戰鬥員`) + 大厅状态(/240/快捷键/版本活动)
    ((("方舟都市", "救援"),
      ("卡厄思", "出擊", "出击", "出擎", "模擬", "模拟", "模凝", "戰鬥員", "战斗员"),
      ("/240", "240", "F1", "F2", "F3", "Seasonal", "機率提升", "机率提升", "等級", "等级")), SceneType.HOME_LOBBY,
     "游戏主界面大厅 -> 点击卡厄思(按键4)"),
)


def text_locked_scene(text_corpus: str):
    """
    L1 文字锁定：命中返回 `(SceneType, 说明)`；未命中返回 None。

    调用方应把它当作**最高优先级的场景判据**（先于打分池）。
    用 `core/text_match.lock_hit` 做模糊 + 组合匹配（OCR 会掉字，不能精确匹配）。
    """
    from ..core.text_match import lock_hit
    # 彻底过滤掉可能残留在 text_corpus 中的 bot 自身调试信息 / 路由标签
    tokens = [
        t for t in (text_corpus or "").split(" ")
        if t and "路由" not in t and not (t.startswith("【") and ("分" in t or "%" in t or "％" in t))
    ]
    for groups, scene, note in TEXT_LOCK_RULES:
        hits = lock_hit(tokens, groups)
        if hits:
            detail = "、".join(f"{g}≈{got}" for g, got in hits)
            return scene, f"{note}｜命中 {detail}"
    return None

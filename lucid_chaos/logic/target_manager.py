"""
Phase 8: Centralized TargetManager
Unifies card and equipment target matching, acquisition tracking, and completion checks.
"""

from typing import List, Dict, Set, Optional, Tuple
import logging

logger = logging.getLogger("TargetManager")

class TargetManager:
    def __init__(self, target_cards: Optional[List[str]] = None, target_equipments: Optional[List[str]] = None):
        self.target_cards: Set[str] = set(target_cards or [])
        self.target_equipments: Set[str] = set(target_equipments or [])
        
        # Track acquired items
        self.acquired_cards: Set[str] = set()
        self.acquired_equipments: Set[str] = set()

    def is_card_target(self, card_name: str) -> bool:
        return any(t in card_name for t in self.target_cards)

    def is_equipment_target(self, equip_name: str) -> bool:
        return any(t in equip_name for t in self.target_equipments)

    def check_matches(self, detected_texts: List[str]) -> Tuple[List[str], List[str]]:
        """Find matching cards and equipments from detected texts."""
        matched_cards = []
        matched_equips = []
        for text in detected_texts:
            for c in self.target_cards:
                if c in text and c not in matched_cards:
                    matched_cards.append(c)
            for e in self.target_equipments:
                if e in text and e not in matched_equips:
                    matched_equips.append(e)
        return matched_cards, matched_equips

    def mark_acquired(self, item_name: str, item_type: str = "auto"):
        if item_type in ("card", "auto") and self.is_card_target(item_name):
            for t in self.target_cards:
                if t in item_name:
                    self.acquired_cards.add(t)
                    logger.info(f"Acquired target card: [{t}]")

        if item_type in ("equipment", "auto") and self.is_equipment_target(item_name):
            for t in self.target_equipments:
                if t in item_name:
                    self.acquired_equipments.add(t)
                    logger.info(f"Acquired target equipment: [{t}]")

    def is_complete(self) -> bool:
        """Returns True if all configured target cards and equipments are acquired."""
        cards_ok = self.acquired_cards.issuperset(self.target_cards)
        equips_ok = self.acquired_equipments.issuperset(self.target_equipments)
        return cards_ok and equips_ok

    def missing_targets(self) -> Dict[str, List[str]]:
        return {
            "cards": list(self.target_cards - self.acquired_cards),
            "equipments": list(self.target_equipments - self.acquired_equipments)
        }

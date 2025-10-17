from typing import Dict

class SinhalaNumberConverter:
    def __init__(self):
        self.positions: Dict[int, dict] = {
            0: {"noun": "", "power_of_ten": 0, "prefix": ""},
            2: {"noun": "සියය", "power_of_ten": 2, "prefix": "සිය"},
            3: {"noun": "දහස", "power_of_ten": 3, "prefix": "දහස්"},
            5: {"noun": "ලක්ෂය", "power_of_ten": 5, "prefix": "ලක්ෂ"},
            6: {"noun": "මිලියනය", "power_of_ten": 6, "prefix": "මිලියන"},
            7: {"noun": "කෝටිය", "power_of_ten": 7, "prefix": "කෝටි"},
            9: {"noun": "බිලියනය", "power_of_ten": 9, "prefix": "බිලියන"},
            12: {"noun": "ත්‍රිලියනය", "power_of_ten": 12, "prefix": "ත්‍රිලියන"},
            15: {"noun": "ක්වාඩ්‍රිලියනය", "power_of_ten": 15, "prefix": "ක්වාඩ්‍රිලියන"},
            18: {"noun": "ක්වින්ටිලියනය", "power_of_ten": 18, "prefix": "ක්වින්ටිලියන"},
            21: {"noun": "සෙක්ස්ටිලියනය", "power_of_ten": 21, "prefix": "සෙක්සිටිලියන"},
            24: {"noun": "සෙප්ටිලියනය", "power_of_ten": 24, "prefix": "සෙප්ටිලියන"},
            27: {"noun": "ඔක්ටිලියනය", "power_of_ten": 27, "prefix": "ඔක්ටිලියන"},
            30: {"noun": "නොලියනය", "power_of_ten": 30, "prefix": "නොලියන"},
            33: {"noun": "ඩෙසිලියනය", "power_of_ten": 33, "prefix": "ඩෙසිලියන"}
        }
        
        self.numbers: Dict[int, dict] = {
            0: {
                "noun": "බිංදුව", "number": 0, "plus_ten_noun": "දහය",
                "plus_ten_prefix": "දස", "prefix": "",
                "product_ten_noun": "බිංදුව", "product_ten_prefix": ""
            },
            1: {
                "noun": "එක", "number": 1, "plus_ten_noun": "එකොළහ",
                "plus_ten_prefix": "එකොළොස්", "prefix": "එක්",
                "product_ten_noun": "දහය", "product_ten_prefix": "දස"
            },
            2: {
                "noun": "දෙක", "number": 2, "plus_ten_noun": "දොළහ",
                "plus_ten_prefix": "දොළොස්", "prefix": "දෙ",
                "product_ten_noun": "විස්ස", "product_ten_prefix": "විසි"
            },
            3: {
                "noun": "තුන", "number": 3, "plus_ten_noun": "දහතුන",
                "plus_ten_prefix": "දහතුන්", "prefix": "තුන්",
                "product_ten_noun": "තිහ", "product_ten_prefix": "තිස්"
            },
            4: {
                "noun": "හතර", "number": 4, "plus_ten_noun": "දහ හතර",
                "plus_ten_prefix": "දහ හතර", "prefix": "හාර",
                "product_ten_noun": "හතළිහ", "product_ten_prefix": "හතළිස්"
            },
            5: {
                "noun": "පහ", "number": 5, "plus_ten_noun": "පහළොව",
                "plus_ten_prefix": "පහළොස්", "prefix": "පන්",
                "product_ten_noun": "පනහ", "product_ten_prefix": "පනස්"
            },
            6: {
                "noun": "හය", "number": 6, "plus_ten_noun": "දහසය",
                "plus_ten_prefix": "දහසය", "prefix": "හය",
                "product_ten_noun": "හැට", "product_ten_prefix": "හැට"
            },
            7: {
                "noun": "හත", "number": 7, "plus_ten_noun": "දහ හත",
                "plus_ten_prefix": "දහ හත්", "prefix": "හත්",
                "product_ten_noun": "හැත්තෑව", "product_ten_prefix": "හැත්තෑ"
            },
            8: {
                "noun": "අට", "number": 8, "plus_ten_noun": "දහ අට",
                "plus_ten_prefix": "දහ අට", "prefix": "අට",
                "product_ten_noun": "අසූව", "product_ten_prefix": "අසූ"
            },
            9: {
                "noun": "නවය", "number": 9, "plus_ten_noun": "දහ නවය",
                "plus_ten_prefix": "දහ නව", "prefix": "නව",
                "product_ten_noun": "අනූව", "product_ten_prefix": "අනූ"
            }
        }
    
    def convert(self, translate_me: int, last_prefix: bool = False) -> str:
        """Convert a number to Sinhala text"""
        if translate_me == 0:
            return self.numbers[0]["noun"]
        
        translated = ""
        powers = sorted(self.positions.keys(), reverse=True)
        last_moded_index = 0
        
        while translate_me > 0:
            position = self.positions[powers[last_moded_index]]
            power_of_ten = position["power_of_ten"]
            
            divided = translate_me // (10 ** power_of_ten)
            moded = translate_me % (10 ** power_of_ten)
            
            prefix = last_prefix or (moded != 0 or power_of_ten != 0)
            translate_me = moded
            
            if divided > 99:
                translated += self.convert(divided, True)
            elif divided > 19:
                divided_by_ten = divided // 10
                moded_by_ten = divided % 10
                
                if moded_by_ten >= 1:
                    translated += self.numbers[divided_by_ten]["product_ten_prefix"] + ' '
                else:
                    if prefix:
                        translated += self.numbers[divided_by_ten]["product_ten_prefix"] + ' '
                    else:
                        translated += self.numbers[divided_by_ten]["product_ten_noun"]
                
                if moded_by_ten >= 1:
                    if prefix:
                        translated += self.numbers[moded_by_ten]["prefix"] + ' '
                    else:
                        translated += self.numbers[moded_by_ten]["noun"]
            
            elif divided > 9:
                moded_by_ten = divided % 10
                if prefix:
                    translated += self.numbers[moded_by_ten]["plus_ten_prefix"] + ' '
                else:
                    translated += self.numbers[moded_by_ten]["plus_ten_noun"]
            
            elif divided > 0 and not (divided == 1 and power_of_ten != 0 and moded == 0 and translated == ''):
                if prefix:
                    translated += self.numbers[divided]["prefix"] + ' '
                else:
                    translated += self.numbers[divided]["noun"]
            
            if divided > 0:
                if moded > 0:
                    translated += position["prefix"] + ' '
                else:
                    translated += position["noun"]
            
            last_moded_index += 1
        
        return translated.strip()

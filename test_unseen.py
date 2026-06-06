import unittest
from src.automata import ProbabilisticAutomata, calculate_levenshtein

class TestUnseenPattern(unittest.TestCase):
    
    def test_levenshtein_distance(self):
        """Levenshtein (Edit Distance) algoritmasının temel hesaplamalarını test eder."""
        print("\n[TEST] Levenshtein Mesafe Algoritması Kontrol Ediliyor...")
        self.assertEqual(calculate_levenshtein("aab", "aab"), 0) # Aynı kelimeler (mesafe 0)
        self.assertEqual(calculate_levenshtein("adc", "abc"), 1) # d -> b değişimi (1 işlem)
        self.assertEqual(calculate_levenshtein("abc", "def"), 3) # Tamamen farklı (3 işlem)
        
    def test_automata_unseen_handling(self):
        """Otomata modelinin unseen (görülmemiş) pattern'ları en yakına eşlemesini test eder."""
        print("[TEST] Otomata Unseen (Görülmemiş) Pattern Eşlemesi Kontrol Ediliyor...")
        
        # Otomata modelini başlat
        automata = ProbabilisticAutomata(window_size=3, alphabet_size=3)
        
        # Manuel olarak bilinen durumlar (known patterns) ekleyelim (Eğitilmiş gibi davranıyoruz)
        automata.known_patterns = {"aab", "abc", "bcc"}
        
        # Sisteme test aşamasında daha önce hiç GÖRMEDİĞİ bir pattern ("adc") verelim [cite: 55]
        unseen_pattern = "adc"
        
        # Unseen fonksiyonunu çalıştır
        mapped_pattern, distance = automata.handle_unseen(unseen_pattern)
        
        # Beklenti: "adc" pattern'ı Levenshtein algoritmasına göre "abc"ye daha yakındır (mesafe: 1)[cite: 56].
        # "aab"ye mesafesi 2, "bcc"ye mesafesi 3'tür. 
        # Bu yüzden sistemin çökmeden bunu "abc"ye eşlemesi zorunludur. [cite: 57]
        self.assertEqual(mapped_pattern, "abc")
        self.assertEqual(distance, 1)
        print(f"-> BAŞARILI: '{unseen_pattern}' pattern'ı başarıyla '{mapped_pattern}' pattern'ına eşlendi (Mesafe: {distance}).")

if __name__ == '__main__':
    unittest.main(verbosity=2)
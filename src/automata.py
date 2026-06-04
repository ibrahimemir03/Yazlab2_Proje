import numpy as np
from collections import defaultdict

def calculate_levenshtein(s1, s2):
    """
    Unseen pattern'lar için Levenshtein (Edit Distance) algoritması.
    İki metin arasındaki (silme, ekleme, değiştirme) mesafe sayısını bulur.
    """
    matrix = np.zeros((len(s1) + 1, len(s2) + 1))
    for i in range(len(s1) + 1):
        matrix[i][0] = i
    for j in range(len(s2) + 1):
        matrix[0][j] = j
        
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i-1] == s2[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      # Silme
                    matrix[i][j-1] + 1,      # Ekleme
                    matrix[i-1][j-1] + 1     # Değiştirme
                )
    return int(matrix[-1][-1])

class ProbabilisticAutomata:
    def __init__(self, window_size=4, alphabet_size=3):
        self.window_size = window_size
        self.alphabet_size = alphabet_size
        
        # Olasılıkları ve geçişleri tutacağımız sözlükler
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.transition_probs = defaultdict(dict)
        self.known_patterns = set()
        
        # 'a', 'b', 'c' gibi alfabemizi oluşturuyoruz
        self.alphabet = [chr(i) for i in range(97, 97 + alphabet_size)]
        
    def _paa(self, data):
        """Piecewise Aggregate Approximation (PAA). Veriyi küçültür."""
        # Veriyi window_size kadar eşit parçaya bölüp ortalamalarını alırız
        splitted = np.array_split(data, self.window_size)
        return np.array([np.mean(x) for x in splitted])

    def _sax(self, paa_data):
        """Symbolic Aggregate approXimation (SAX). Sayıları harflere çevirir."""
        # Veriyi alphabet_size kadar yüzdelik dilime (bin) ayırıyoruz
        bins = np.percentile(paa_data, np.linspace(0, 100, self.alphabet_size + 1)[1:-1])
        indices = np.digitize(paa_data, bins)
        return "".join([self.alphabet[i] for i in indices])

    def fit(self, data):
        """Eğitim verisi üzerinden state ve geçişleri öğrenir."""
        patterns = []
        
        # 1. Sliding Window ile tüm patternleri çıkar
        for i in range(len(data) - self.window_size + 1):
            window = data[i : i + self.window_size]
            paa_val = self._paa(window)
            sax_str = self._sax(paa_val)
            
            patterns.append(sax_str)
            self.known_patterns.add(sax_str)
            
        # 2. Geçişleri (Transitions) Sayma
        for i in range(len(patterns) - 1):
            current_state = patterns[i]
            next_state = patterns[i+1]
            self.transitions[current_state][next_state] += 1
            
        # 3. Geçiş Olasılıklarını Hesaplama
        for state, next_states in self.transitions.items():
            total_out = sum(next_states.values())
            for n_state, count in next_states.items():
                self.transition_probs[state][n_state] = count / total_out

    def handle_unseen(self, pattern):
        """
        Unseen pattern gelirse Levenshtein ile en yakın state'i bulur.
        Döndürdüğü: (En_Yakin_Pattern, Mesafe)
        """
        if pattern in self.known_patterns:
            return pattern, 0
            
        min_dist = float('inf')
        closest_pattern = None
        
        for known in self.known_patterns:
            dist = calculate_levenshtein(pattern, known)
            if dist < min_dist:
                min_dist = dist
                closest_pattern = known
                
        return closest_pattern, min_dist
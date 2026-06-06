import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
from src.data_loader import load_config, load_batadal_data
from src.preprocessor import TimeSeriesPreprocessor
from src.automata import ProbabilisticAutomata

def generate_automata_visuals():
    print("Hocanın istediği eksik grafikler çiziliyor...")
    os.makedirs('models', exist_ok=True)
    config = load_config("config.json")
    
    # Hızlıca eğitim verisini simüle edip modeli eğitiyoruz
    batadal_df = load_batadal_data(config['data_paths']['batadal'])
    prep = TimeSeriesPreprocessor()
    train_df = batadal_df.iloc[:int(len(batadal_df)*0.6)]
    train_processed = prep.fit_transform(train_df, ['DATETIME', 'ATT_FLAG'])
    
    # Grafiğin aşırı kalabalık olup okunmaz hale gelmemesi için 
    # küçük bir pencere/alfabe boyutu seçiyoruz
    automata = ProbabilisticAutomata(window_size=3, alphabet_size=3)
    automata.fit(train_processed['PC1'].values)
    
    # Benzersiz durumları (states) al ve sınırla (Görsellik için en sık geçen 15 tanesi)
    states = list(automata.known_patterns)
    if len(states) > 15: 
        states = states[:15]
    
    # --- 1. Transition Probability Heatmap (Geçiş Olasılıkları Isı Haritası) ---
    mat = pd.DataFrame(0.0, index=states, columns=states)
    for s1 in states:
        for s2 in states:
            mat.loc[s1, s2] = automata.transition_probs.get(s1, {}).get(s2, 0.0)
            
    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Transition Probability Heatmap")
    plt.tight_layout()
    plt.savefig("models/transition_heatmap.png")
    plt.close()
    print("1. Heatmap başarıyla 'models/transition_heatmap.png' olarak kaydedildi.")
    
    # --- 2. Automata State Diagram (Durum Geçiş Diyagramı) ---
    G = nx.DiGraph()
    for s1 in states:
        for s2, prob in automata.transition_probs.get(s1, {}).items():
            if s2 in states and prob > 0.05: # Sadece anlamlı (olasılığı %5'ten büyük) geçişleri çiz
                G.add_edge(s1, s2, weight=prob)
    
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2500, font_size=10, font_weight='bold', arrows=True, arrowsize=20)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=9)
    plt.title("Automata State Diagram")
    plt.savefig("models/state_diagram.png")
    plt.close()
    print("2. State Diagram başarıyla 'models/state_diagram.png' olarak kaydedildi.")

if __name__ == "__main__":
    generate_automata_visuals()
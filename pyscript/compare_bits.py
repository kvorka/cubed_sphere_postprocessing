import os
import numpy as np

def porovnej_single_tile(soubor1, soubor2, n_stena=32):
    """
    Porovná dva binární soubory pro jednu konkrétní dlaždici (tile).
    Rozměr pole pro jednu dlaždici je (n+1) x (n+1).
    """
    n_bodu = n_stena + 1
    velikost_pole = n_bodu * n_bodu
    pocet_poli = 16
    ocekavana_velikost = pocet_poli * velikost_pole * 8  # float64 = 8 bajtů

    nazvy_poli = [
        "XC", "YC", "DXF", "DYF", "RAC", "XG", "YG", "DXV", 
        "DYU", "RAZ", "DXC", "DYC", "RAW", "RAS", "DXG", "DYG"
    ]

    # Kontrola existence a velikosti
    for s in [soubor1, soubor2]:
        if not os.path.exists(s):
            return f"❌ Soubor '{s}' neexistuje."
        if os.path.getsize(s) != ocekavana_velikost:
            return f"⚠️ Soubor '{s}' má nesprávnou velikost ({os.path.getsize(s)} B, očekáváno {ocekavana_velikost} B)."

    # Načtení dat (big-endian float64)
    data1 = np.fromfile(soubor1, dtype='>f8').reshape(pocet_poli, n_bodu, n_bodu)
    data2 = np.fromfile(soubor2, dtype='>f8').reshape(pocet_poli, n_bodu, n_bodu)

    if np.array_equal(data1, data2):
        return "✅ OK (Identické)"

    # Pokud se liší, detailně analyzujeme které pole
    vypis_rozdilu = []
    for i in range(pocet_poli):
        pole1 = data1[i]
        pole2 = data2[i]
        
        if not np.array_equal(pole1, pole2):
            rozdil = np.abs(pole1 - pole2)
            max_rozdil = np.max(rozdil)
            idx_y, idx_x = np.unravel_index(np.argmax(rozdil), rozdil.shape)
            
            vypis_rozdilu.append(
                f"   -> Pole [{i:02d}] {nazvy_poli[i]}: "
                f"max rozdíl {max_rozdil:.2e} "
                f"(Val1: {pole1[idx_y, idx_x]:.4f}, Val2: {pole2[idx_y, idx_x]:.4f}) "
                f"rel rozdíl {np.abs(max_rozdil/pole1[idx_y, idx_x]):.2e}"
            )
            
    return "\n".join(vypis_rozdilu)


def porovnej_vsechny_tiles(adresar_a, adresar_b, pocet_tiles=6, n_stena=32):
    """
    Projde zadané adresáře a porovná soubory tile001.mitgrid až tile00X.mitgrid.
    """
    print(f"=== Porovnání sad dlaždic (rozlišení {n_stena}x{n_stena}) ===")
    print(f"Sada A: {adresar_a}")
    print(f"Sada B: {adresar_b}\n" + "-"*50)

    for i in range(1, pocet_tiles + 1):
        jmeno_souboru = f"tile{i:03d}.mitgrid"
        cesta_a = os.path.join(adresar_a, jmeno_souboru)
        cesta_b = os.path.join(adresar_b, jmeno_souboru)
        
        print(f"Porovnávám: {jmeno_souboru}")
        vysledek = porovnej_single_tile(cesta_a, cesta_b, n_stena)
        print(vysledek)
        print("-" * 50)

# --- Spuštění ---
if __name__ == "__main__":
    # PŘÍKLAD 1: Porovnání celých složek (předpokládá se 6 stěn cubed-sphere)
    slozka_verze1 = "./original_grid"
    slozka_verze2 = "."
    
    porovnej_vsechny_tiles(slozka_verze1, slozka_verze2, pocet_tiles=6, n_stena=32)
    
    # PŘÍKLAD 2: Pokud chcete porovnat jen dva konkrétní soubory napřímo, odkomentujte:
    # vysledek = porovnej_single_tile("cesta/tile001.mitgrid", "druha_cesta/tile001.mitgrid", n_stena=32)
    # print(vysledek)
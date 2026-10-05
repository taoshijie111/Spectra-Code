# Supplementary Encoding Results

These files accompany Supplementary Tables S4-S10 of the second-round revision.

| File | Contents |
| --- | --- |
| `versioned_dictionary_400_1023.jsonl.gz` | One word, computed-spectrum source row, canonical molecular graph, source-member digest, and 400-bin code per line |
| `table_s4.csv` | Formal capacity in bits, occupied codes, entropy, and pair-collision probability |
| `table_s5.csv` | Reconstruction fidelity, test records with codes absent from training, and fixed-cohort molecular readout |
| `table_s6a.csv` | Reconstruction cosine under seven matched perturbations |
| `table_s6b.csv` | Exact-code retention under the same seven perturbations |
| `table_s7.csv` | Reconstruction cosine and exact-code retention under constant and linear baseline variation |
| `table_s8.csv` | Fixed-reader source-graph recovery under seven perturbations |
| `table_s9.csv` | Mean reconstruction fidelity and 2 cm-1 code retention for 40 NIST species |
| `table_s10.csv` | Native-grid reconstruction cosine at five nominal NIST resolutions |

The matched computed-spectrum population contains 127,468 records, split by canonical molecular graph into 102,123 training, 12,794 validation, and 12,551 test records. Tables S4-S7 use this population. The molecular-readout comparison in Table S5 uses the same fixed 1,024-source test cohort for every design. Tables S9-S10 use 40 species with five spectra each from the [NIST Quantitative Infrared Database](https://webbook.nist.gov/chemistry/quant-ir/).

The versioned dictionary uses the same 20,005 normalized labels as the original 10/9 W2C cipher, with a separate 400/1023 assignment. Each JSONL record contains `word`, `source_row`, `canonical_smiles`, `public_member`, `public_member_sha256`, and `code`. Its SHA-256 digest is `403c4fbd034ff776f9a2d138a9a91298b976e9c9ec8a69790876c7756dfa2285`.

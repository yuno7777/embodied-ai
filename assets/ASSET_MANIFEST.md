# Asset manifest

Every visual asset in the observer is locally stored and first-party:

| Asset | Source | Role |
| --- | --- | --- |
| `frontend/assets/floor-tile.png` | Generated for this project | Top-down floor texture |
| `frontend/assets/agent.png` | Generated for this project | Embodied-agent sprite |
| `frontend/assets/exit-key.png` | Generated for this project | Exit-key sprite |
| `frontend/assets/hazard.png` | Generated for this project | Electrical-hazard sprite |
| `frontend/assets/supplies.png` | Generated for this project | Water/ration pickup sprite |
| `frontend/assets/caretaker.png` | Generated for this project | NPC sprite |
| `frontend/observer/public/key-locker.png` | Generated for this project | Interactable key-locker sprite |
| `assets/models/*.obj` | Hand-authored for this project | Future 3D renderer props |

There are deliberately no third-party image, icon, font, or model dependencies. The 2D dashboard uses only the generated PNGs above and native system fonts.

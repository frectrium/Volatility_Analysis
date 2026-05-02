import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shared_eval.eval_grid import LOG_MONEYNESS_GRID, MATURITY_YEARS_GRID
from shared_eval.plots import COLOURS, PIPELINES, _black76_call_vec

def add_new_plots(surfaces, p1_price, payload, out_dir):
    # 8. Failure slice
    if "B" in p1_price and "C" in p1_price:
        days_B = {d["date"]: d["rmse"] for d in p1_price["B"].get("per_day", [])}
        days_C = {d["date"]: d["rmse"] for d in p1_price["C"].get("per_day", [])}
        candidates = [(d, days_B[d], days_C[d]) for d in days_B if d in days_C]
        if candidates:
            candidates.sort(key=lambda x: x[2] - x[1], reverse=True)
            d_str = candidates[0][0]
            d = pd.Timestamp(d_str).normalize()
            quotes = payload["quotes_dict"]
            if d in quotes:
                df = quotes[d]
                target_T = 30 / 365.0
                df_slice = df[(df["maturity_days"] >= 20) & (df["maturity_days"] <= 45)]
                if len(df_slice) > 5:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.scatter(df_slice["log_moneyness"], df_slice["mid"], c="black", s=15, label="market price", alpha=0.6)
                    kg = np.linspace(df_slice["log_moneyness"].min(), df_slice["log_moneyness"].max(), 50)
                    mean_F = df_slice["fwd_price"].mean()
                    mean_disc = df_slice["discount"].mean()
                    K_grid = mean_F * np.exp(kg)
                    from shared_eval.eval_grid import interp_surface_to_points
                    for n in PIPELINES:
                        if n not in surfaces or d not in surfaces[n]: continue
                        ivs = interp_surface_to_points(surfaces[n][d], kg, np.full_like(kg, target_T))
                        prices = _black76_call_vec(mean_F, K_grid, target_T, ivs, mean_disc)
                        ax.plot(kg, prices, color=COLOURS[n], label=n, lw=1.5)
                    ax.set_xlabel("log-moneyness")
                    ax.set_ylabel("Price")
                    ax.set_title(f"Failure Case for C (Price Slice) - {d.date()}")
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(out_dir / f"08_failure_case_price_{d.date()}.png", dpi=150)
                    plt.close(fig)

    # 9. 3D Surfaces
    if surfaces and "A" in surfaces:
        d = list(surfaces["A"].keys())[len(surfaces["A"]) // 2]
        quotes = payload["quotes_dict"]
        if d in quotes:
            df = quotes[d]
            F = float(df["fwd_price"].mean())
            disc = float(df["discount"].mean())
            
            Kg, Tg = np.meshgrid(LOG_MONEYNESS_GRID, MATURITY_YEARS_GRID, indexing='ij')
            K_abs = F * np.exp(Kg)
            
            for n in PIPELINES:
                if n not in surfaces or d not in surfaces[n]: continue
                surf = surfaces[n][d]
                
                # IV 3D
                fig = plt.figure(figsize=(10, 4))
                ax = fig.add_subplot(121, projection='3d')
                ax.plot_surface(Kg, Tg, surf, cmap='viridis', alpha=0.8)
                ax.scatter3D(df["log_moneyness"], df["T"], df["sigma_market"], c='black', marker='o', s=5, alpha=0.5, label='Market')
                ax.set_xlabel('log-moneyness')
                ax.set_ylabel('Maturity (Y)')
                ax.set_zlabel('IV')
                ax.set_title(f'{n} IV Surface')
                
                # Price 3D
                ax2 = fig.add_subplot(122, projection='3d')
                prices = _black76_call_vec(F, K_abs, Tg, surf, disc)
                ax2.plot_surface(Kg, Tg, prices, cmap='plasma', alpha=0.8)
                ax2.scatter3D(df["log_moneyness"], df["T"], df["mid"], c='black', marker='o', s=5, alpha=0.5, label='Market')
                ax2.set_xlabel('log-moneyness')
                ax2.set_ylabel('Maturity (Y)')
                ax2.set_zlabel('Price')
                ax2.set_title(f'{n} Price Surface')
                
                fig.suptitle(f"Pipeline {n} Surfaces - {d.date()}")
                fig.tight_layout()
                fig.savefig(out_dir / f"09_3d_surfaces_{n}.png", dpi=150)
                plt.close(fig)


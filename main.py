
import logging
from datetime import datetime
from backtesting.backtest import Backtest
from utils.pair_finder import PairsFinder
from utils.signal_generator import SignalGenerator
from config.settings import SystemConfig
from data.data_provider import DataProvider
from utils.risk_manager import RiskManager
from backtesting.performance_analyzer import PerformanceAnalyzer
from utils.plotting_utils import Plotter
import pandas as pd


def main_with_analysis():

    config = SystemConfig()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    data_provider = DataProvider(config.data)
    data = data_provider.get_data()
    
    pairs_finder = PairsFinder(config.pairs, config.data.universe)
    pairs = pairs_finder.find_pairs(data)

    backtest_engine = Backtest(
        backtest_config=config.backtest,
        pairs_config=config.pairs,
        risk_config=config.risk,
        transaction_cost_config = config.tranc
    )
    
    signal_generator = SignalGenerator(config.model)
    risk_manager = RiskManager(config.risk)
    
    results = backtest_engine.run_backtest(data, pairs, signal_generator, risk_manager)
    
    if results:
        portfolio_results = results['daily_results']
        trades = results['trades']
        
        trades_df = pd.DataFrame(trades) if isinstance(trades, list) else trades
        
        analyzer = PerformanceAnalyzer(
            results_df=portfolio_results,
            trades_df=trades_df,
            initial_capital=config.risk.initial_capital
        )
        
        analyzer.print_summary()
        metrics = analyzer.calculate_all_metrics()
        analyzer.print_pair_metrics()
        # Generate plots
        # plotter = Plotter(figsize=(15, 10))
        # top_pair = pairs[0]
        # symbol1, symbol2 = top_pair['symbol1'], top_pair['symbol2']
        # fig1 = plotter.plot_portfolio_performance(portfolio_results, config.risk.initial_capital)
        # fig1.savefig('portfolio_performance.png', dpi=300, bbox_inches='tight')
        # price1_full = data[symbol1]
        # price2_full = data[symbol2]
        # logger.info("3. Spread analysis...")
        # spread = price1_full - (top_pair['intercept'] + top_pair['hedge_ratio'] * price2_full)
        
        # fig3 = plotter.plot_spread_analysis(
        #     spread,
        #     top_pair['hedge_ratio'],
        #     top_pair['intercept'],
        #     symbol1,
        #     symbol2
        # )
        # fig3.savefig('plots/03_spread_analysis.png', dpi=300, bbox_inches='tight')
        # logger.info("   Saved: plots/03_spread_analysis.png")
        
        # # 4. Z-Score Signals
        # logger.info("4. Z-score signals...")
        # rolling_mean = spread.rolling(window=30).mean()
        # rolling_std = spread.rolling(window=30).std()
        # z_scores = (spread - rolling_mean) / rolling_std
        
        # fig4 = plotter.plot_zscore_signals(
        #     z_scores,
        #     config.pairs.entry_z_score,
        #     config.pairs.exit_z_score,
        #     symbol1,
        #     symbol2
        # )
        # fig4.savefig('plots/04_zscore_signals.png', dpi=300, bbox_inches='tight')
        # logger.info("   Saved: plots/04_zscore_signals.png")
        
        # # 5. Price Ratio with Signals (FROM REPOSITORY EXAMPLE)
        # logger.info("5. Price ratio with buy/sell signals...")
        # fig5 = plotter.plot_price_ratio_with_signals(
        #     price1_full,
        #     price2_full,
        #     z_scores,
        #     config.pairs.entry_z_score,
        #     symbol1,
        #     symbol2
        # )
        # fig5.savefig('plots/05_price_ratio_signals.png', dpi=300, bbox_inches='tight')
        # logger.info("   Saved: plots/05_price_ratio_signals.png")
        
        # # 6. Rolling Z-Score with Spread (FROM REPOSITORY EXAMPLE)
        # logger.info("6. Rolling z-score analysis...")
        # fig6 = plotter.plot_rolling_zscore(
        #     spread,
        #     window=30,
        #     entry_threshold=config.pairs.entry_z_score,
        #     exit_threshold=config.pairs.exit_z_score
        # )
        # fig6.savefig('plots/06_rolling_zscore.png', dpi=300, bbox_inches='tight')
        # logger.info("   Saved: plots/06_rolling_zscore.png")

        
        # # 8. Cumulative Returns
        # logger.info("8. Cumulative returns...")
        # fig8 = plotter.plot_cumulative_returns(portfolio_results)
        # fig8.savefig('plots/08_cumulative_returns.png', dpi=300, bbox_inches='tight')
        # logger.info("   Saved: plots/08_cumulative_returns.png")
        
        
        # # 10. Rolling Sharpe Ratio
        # logger.info("10. Rolling Sharpe ratio...")
        # fig10 = plotter.plot_rolling_sharpe(portfolio_results, window=60)
        # fig10.savefig('plots/10_rolling_sharpe.png', dpi=300, bbox_inches='tight')
        # logger.info("   Saved: plots/10_rolling_sharpe.png")
        
        # # 12. Correlation Heatmap
        # logger.info("12. Correlation heatmap...")
        # pair_symbols = [(p['symbol1'], p['symbol2']) for p in pairs[:5]]
        # fig12 = plotter.plot_correlation_heatmap(
        #     data[data.columns[:20]],  # Top 20 symbols for readability
        #     pair_symbols
        # )
        # fig12.savefig('plots/12_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        # logger.info("   Saved: plots/12_correlation_heatmap.png")

if __name__ == "__main__":
    main_with_analysis()

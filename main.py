"""
Main execution script for Statistical Arbitrage System
"""
import logging
from backtesting.backtest import Backtest
from config.settings import SystemConfig
from data.data_provider import DataProvider
from utils.pair_finder import PairsFinder
from utils.lstm_kalman import DeepPairsSignalGenerator
from utils.risk_manager import RiskManager


def setup_logging(log_level='INFO'):
    logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    config = SystemConfig()
    setup_logging("WARNING") 
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Statistical Arbitrage System")
    
    data_provider = DataProvider(config.data)
    pairs_finder = PairsFinder(config.pairs)
    signal_generator = DeepPairsSignalGenerator(config.model)
    risk_manager = RiskManager(config.risk)
    backtest_engine = Backtest(config.backtest)
    
    logger.info("Loading data...")
    data = data_provider.get_data()
    
    logger.info("Finding pairs...")
    pairs = pairs_finder.find_pairs(data)
    
    if not pairs:
        logger.error("No valid pairs found")
        return
    
    logger.info(f"Found {len(pairs)} valid pairs")
    
    logger.info("Running backtest...")
    results = backtest_engine.run_backtest(data, pairs, signal_generator, risk_manager)
    print(results['overall_performance'])

    pair_performance = results['pair_performance']
    for pair_name, stats in pair_performance.items():
        print(f"--- Performance for Pair: {pair_name} ---")
        
        # Access and print each metric
        print(f"  Sharpe Ratio: {stats.get('sharpe_ratio', 'N/A'):.2f}")
        print(f"  Max Drawdown: {stats.get('max_drawdown', 'N/A') * 100:.2f}%")
        print(f"  Total PnL: ${stats.get('total_pnl', 0):,.2f}")
        print(f"  Win Rate: {stats.get('win_rate', 0) * 100:.2f}%")
        print(f"  Total Trades: {stats.get('total_trades', 0)}")
        print("-" * 35 + "\n")
    return results

if __name__ == "__main__":
    results = main()

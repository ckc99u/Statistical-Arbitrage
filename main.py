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
    setup_logging(config.log_level)
    
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
    results = backtest_engine.run_backtest(data, pairs[:10], signal_generator, risk_manager)  # Limit to top 10 pairs
    
    logger.info("Backtest Results:")
    for metric, value in results['performance'].items():
        if isinstance(value, float):
            logger.info(f"{metric}: {value:.4f}")
        else:
            logger.info(f"{metric}: {value}")
    
    return results

if __name__ == "__main__":
    results = main()

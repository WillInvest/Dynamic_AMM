from env.market import MarketSimulator
from env.new_amm import AMM
from env.multiAmm import MultiAgentAmm

market = MarketSimulator()
amm = AMM()
env = MultiAgentAmm(market=market, amm=amm)
obs, _ = env.reset()

print(obs)
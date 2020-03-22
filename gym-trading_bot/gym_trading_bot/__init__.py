from gym.envs.registration import register

register(id='TradingBot-v0',
entry_point='gym_trading_bot.envs:TradingBotEnv')
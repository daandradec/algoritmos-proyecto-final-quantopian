import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import SimpleMovingAverage

from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline.data import morningstar as mstar

from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.data import Fundamentals

from quantopian.pipeline.factors import Returns

# Constraint Parameters
MAX_GROSS_LEVERAGE = 1.0
TOTAL_POSITIONS = 300

MAX_SHORT_POSITION_SIZE = 2.0 / TOTAL_POSITIONS
MAX_LONG_POSITION_SIZE = 2.0 / TOTAL_POSITIONS

def initialize(context):
    
    algo.attach_pipeline(make_pipeline(), 'long_short_equity_template')
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_factors')
    algo.schedule_function(func=rebalance,
                           date_rule=algo.date_rules.week_start(),
                           time_rule=algo.time_rules.market_open(hours=0, minutes=30),
                           half_days=True)
    algo.schedule_function(func=record_vars,
                           date_rule=algo.date_rules.every_day(),
                           time_rule=algo.time_rules.market_close(),
                           half_days=True)


def make_pipeline():
    universe = QTradableStocksUS()
    
     # Variables seleccionadas del dataframe de Fundamentals
    value = Fundamentals.ebit.latest / Fundamentals.enterprise_value.latest
    quality = Fundamentals.roe.latest
    sentiment_score = SimpleMovingAverage(inputs=[stocktwits.bull_minus_bear],window_length=3,)    
    buildings_and_improvements = Fundamentals.buildings_and_improvements.latest 
    account_payable = Fundamentals.accounts_payable.latest
    current_assets = Fundamentals.current_assets.latest
    total_equity = Fundamentals.total_equity.latest
    accumulated_depreciation = Fundamentals.accumulated_depreciation.latest
    diluted_eps_growth = Fundamentals.diluted_eps_growth.latest
    dps_growth = Fundamentals.dps_growth.latest

    # Variables (SIN ATIPICOS)
    value_winsorized = value.winsorize(min_percentile=0.05, max_percentile=0.95)
    quality_winsorized = quality.winsorize(min_percentile=0.05, max_percentile=0.95)
    sentiment_score_winsorized = sentiment_score.winsorize(min_percentile=0.05,max_percentile=0.95)
    buildings_and_improvements = buildings_and_improvements.winsorize(min_percentile=0.05,max_percentile=0.95)
    account_payable = account_payable.winsorize(min_percentile=0.05,max_percentile=0.95)
    current_assets = current_assets.winsorize(min_percentile=0.05,max_percentile=0.95)
    total_equity = total_equity.winsorize(min_percentile=0.05,max_percentile=0.95)
    accumulated_depreciation = accumulated_depreciation.winsorize(min_percentile=0.05,max_percentile=0.95)
    diluted_eps_growth = diluted_eps_growth.winsorize(min_percentile=0.05,max_percentile=0.95)
    dps_growth = dps_growth.winsorize(min_percentile=0.05,max_percentile=0.95)
    
    
    # FACTOR COMBINADO
    combined_factor = (
        quality_winsorized.zscore()*0.2+
        value_winsorized.zscore()*0.4+
        dps_growth.zscore()*0.15 +
        sentiment_score_winsorized.zscore()*0.3+
        buildings_and_improvements.zscore()*0.2+
        accumulated_depreciation.zscore()*0.25 +
        account_payable.zscore()*0.01 +
        current_assets.zscore()*0.01 +
        total_equity.zscore()*0.01 +
        diluted_eps_growth.zscore()*0.01
    )

    longs = combined_factor.top(TOTAL_POSITIONS//2, mask=universe)
    shorts = combined_factor.bottom(TOTAL_POSITIONS//2, mask=universe)

    long_short_screen = (longs | shorts)

    pipe = Pipeline(
        columns={
            'longs': longs,
            'shorts': shorts,
            'combined_factor': combined_factor
        },
        screen=long_short_screen
    )
    return pipe


def before_trading_start(context, data):
    context.pipeline_data = algo.pipeline_output('long_short_equity_template')
    context.risk_loadings = algo.pipeline_output('risk_factors')


def record_vars(context, data):
    algo.record(num_positions=len(context.portfolio.positions))


# Called at the start of every month in order to rebalance
# the longs and shorts lists
def rebalance(context, data):
    pipeline_data = context.pipeline_data

    risk_loadings = context.risk_loadings
    objective = opt.MaximizeAlpha(pipeline_data.combined_factor)
    constraints = []
    constraints.append(opt.MaxGrossExposure(MAX_GROSS_LEVERAGE))
    constraints.append(opt.DollarNeutral())
    neutralize_risk_factors = opt.experimental.RiskModelExposure(
        risk_model_loadings=risk_loadings,
        version=0
    )
    constraints.append(neutralize_risk_factors)
    constraints.append(
        opt.PositionConcentration.with_equal_bounds(
            min=-MAX_SHORT_POSITION_SIZE,
            max=MAX_LONG_POSITION_SIZE
        ))
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints
    )
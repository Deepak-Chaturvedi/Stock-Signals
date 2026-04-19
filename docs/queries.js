// queries.js

// SELECT DISTINCT A.Symbol, B.COMPANY_NAME AS Name,
//   "Accumulation" AS 'Signal Type',
//   DATE(Date) AS 'Signal Date'
//   FROM SIGNAL_ACCUMULATION_STEADY AS A
//   LEFT JOIN STOCK_DETAILS AS B
//     ON A.Symbol = B.Symbol
//   WHERE B.EXCHANGE != 'BSE'
//   ORDER BY "Signal Date" DESC, A.AD_Slope DESC, A.Avg_Volume_Spike DESC;

window.QUERY_SIGNAL_ACCUMULATION = `
  SELECT DISTINCT 
    A.Symbol, 
    B.COMPANY_NAME AS 'Company Name',
    A.Signal_Type AS 'Signal Type',
    DATE(A.Signal_date) AS 'Signal Date',

    CAST(ROUND(A.Updated_Signal_Price,0) AS INTEGER) AS 'Signal Price',
    CAST(ROUND(A.current_price,0) AS INTEGER) AS 'Current Price',

    -- 🎯 Reality
    CAST(ROUND(A.ret_sinceSignal,0) AS INTEGER) || '%' AS 'Current Return %',

    -- 🚀 Best Case
    CAST(ROUND(A.ret_1w_max,0) AS INTEGER) || '%' AS '1W Best %',
    CAST(ROUND(A.ret_2w_max,0) AS INTEGER) || '%' AS '2W Best %',
    CAST(ROUND(A.ret_1m_max,0) AS INTEGER) || '%' AS '1M Best %',

    -- 🔻 Risk
    CAST(ROUND(A.ret_sinceSignal_dd,0) AS INTEGER) || '%' AS 'Max Drawdown %',

    -- ⚡ Behavior
    A.ret_1m_time_to_peak AS 'Days to Peak',
    CAST(ROUND(A.ret_1m_peak_to_end,0) AS INTEGER) || '%' AS 'From Peak %',
    ROUND(A.ret_1m_pct_days_profit * 100,0) || '%' AS 'Days in Profit %'

  FROM SIGNAL_RETURNS AS A
  LEFT JOIN STOCK_DETAILS AS B
    ON A.Symbol = B.Symbol

  WHERE B.EXCHANGE != 'BSE'
  AND B.UPDATE_DATE = (SELECT MAX(UPDATE_DATE) FROM STOCK_DETAILS)

  ORDER BY DATE(A.Signal_date) DESC, A.Signal_Rank ASC
  ;
`;
// script.js
//ORDER BY  7 DESC,8 DESC,9 DESC,10 DESC,11 DESC,12 DESC,13 DESC

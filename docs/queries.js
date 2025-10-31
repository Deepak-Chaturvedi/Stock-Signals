// queries.js

// SELECT DISTINCT A.Symbol, B.COMPANY_NAME AS Name,
//   "Accumulation Signal" AS 'Signal Type',
//   DATE(Date) AS 'Signal Date'
//   FROM SIGNAL_ACCUMULATION_STEADY AS A
//   LEFT JOIN COMPANY_DETAILS AS B
//     ON A.Symbol = B.Symbol
//   WHERE B.EXCHANGE != 'BSE'
//   ORDER BY "Signal Date" DESC, A.AD_Slope DESC, A.Avg_Volume_Spike DESC;

window.QUERY_SIGNAL_ACCUMULATION = `
SELECT DISTINCT 
  A.Symbol, 
  B.COMPANY_NAME AS Name,
  a.Signal_Type as 'Signal Type',
  DATE(A.Signal_date) AS 'Signal Date',
  CAST(ROUND(a.Signal_Price,0) AS INTEGER) AS 'Signal Price',
  CAST(ROUND(current_price, 0) AS INTEGER) AS 'Current Price',
  cast(round(ret_1w_perc  ,0)as INTEGER) || '%' AS '1 Week Return %',
  cast(round(ret_2w_perc  ,0)as INTEGER) || '%' AS '2 Week Return %',
  cast(round(ret_1m_perc  ,0)as INTEGER) || '%' AS '1 Month Return %',
  cast(round(ret_3m_perc  ,0)as INTEGER) || '%' AS '3 Month Return %',
  cast(round(ret_6m_perc  ,0)as INTEGER) || '%' AS '6 Month Return %',
  cast(round(ret_1y_perc  ,0)as INTEGER) || '%' AS '1 Year Return %',
  cast(round(ret_sinceSignal_perc ,0) AS INTEGER) || '%' AS 'Return Since Signal %'
  FROM SIGNAL_RETURNS AS A
  LEFT JOIN COMPANY_DETAILS AS B
    ON A.Symbol = B.Symbol
  WHERE B.EXCHANGE != 'BSE'
  ORDER BY DATE(A.Signal_date) DESC, A.Signal_Rank ASC
  ;
`;
// script.js
//ORDER BY  7 DESC,8 DESC,9 DESC,10 DESC,11 DESC,12 DESC,13 DESC

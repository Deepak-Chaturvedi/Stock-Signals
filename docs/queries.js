// queries.js

window.QUERY_SIGNAL_ACCUMULATION = `  SELECT DISTINCT A.Symbol, B.COMPANY_NAME AS Name,
  "Accumulation Signal" AS 'Signal Type',
  DATE(Date) AS 'Signal Date'
  FROM SIGNAL_ACCUMULATION_STEADY AS A
  LEFT JOIN COMPANY_DETAILS AS B
    ON A.Symbol = B.Symbol
  WHERE B.EXCHANGE != 'BSE'
  ORDER BY "Signal Date" DESC, A.AD_Slope DESC, A.Avg_Volume_Spike DESC;
`;

// script.js
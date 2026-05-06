from fastmcp import FastMCP
from finance import FinanceKG

mcp = FastMCP(
    name="Finance KG MCP Server",
    port=8081,
    host="0.0.0.0",
    log_level="DEBUG",
    on_duplicate_tools="warn"
)
kg = FinanceKG()

@mcp.resource(
    uri="metadata://finance-company-format",
    name="finance_format_example",
    description="A reference resource that lists well-formatted company name and ticker symbol pairs. Intended to guide users and tools in handling company-related finance queries accurately.",
    mime_type="text/plain",
)
def get_company_format_examples() -> str:
    """
    Provides example entries of valid company names and their corresponding ticker symbols.  
    Useful for understanding the expected format and guiding company name processing tools.
    """
    return "\n".join([
        "Applied Optoelectronics Inc. Common Stock,AAOI",
        "Airbnb Inc. Class A Common Stock,ABNB",
        "Tesla Inc. Common Stock,TSLA",
        "Microsoft Corporation Common Stock,MSFT",
        "Evolent Health Inc Class A Common Stock,EVH"
    ])

@mcp.tool()
def match_ticker_or_name(query: str) -> str:
    """
    Checks if the given input is a valid ticker symbol or a company name.  
    If it's a valid ticker, returns the corresponding company name.  
    If it's a valid company name, returns the corresponding ticker.  
    Otherwise, returns an appropriate error message.

    arg:
        query: str (a short string representing either a single ticker symbol or full company name)
    output:
        A message indicating whether the input matches a known ticker or company, and the corresponding pair.
    """
    # Normalize input
    query_norm = query.strip()

    # Load the company dictionary from the KG (assumes name_dict: {Name: Symbol})
    # Reverse dict for symbol to name lookup
    name_to_symbol = kg.name_dict
    symbol_to_name = {v.upper(): k for k, v in name_to_symbol.items()}

    # Case 1: input is a ticker symbol
    if query_norm.upper() in symbol_to_name:
        company_name = symbol_to_name[query_norm.upper()]
        return f'"{query}" is recognized as a ticker symbol.\nCompany Name: {company_name}\nTicker Symbol: {query_norm.upper()}'

    # Case 2: input is a company name
    # Do case-insensitive matching
    matches = [name for name in name_to_symbol if name.lower() == query.lower()]
    if matches:
        company_name = matches[0]
        ticker = name_to_symbol[company_name]
        return f'"{query}" is recognized as a company name.\nCompany Name: {company_name}\nTicker Symbol: {ticker}'

    return f'"{query}" is not recognized as either a valid company name or ticker symbol in the Finance KG.'

@mcp.tool()
def get_company_name(query: str):
    """
    Retrieves candidate company names matching a natural language query.  
    Input must not be a ticker symbol (e.g., 'AAPL', 'MSFT'), but a descriptive name (e.g., 'Apple', 'Microsoft').  
    This tool is typically used as the first step before resolving a company's ticker symbol.
    
    arg:
        query: str
    output:
        top matched company names: list[str]
    """
    return kg.get_company_name(query)

@mcp.tool()
def get_ticker_by_name(company_name: str):
    """
    Retrieves the ticker symbol associated with a specific company name.  
    The input should be a full company name, preferably obtained via get_company_name.  
    Do not pass ticker-like strings (e.g., 'TSLA', 'MUJ') directly to this tool.
    
    arg:
        company_name: the company name: str
    output:
        the ticker name of the company: str
    """
    return kg.get_ticker_by_name(company_name)

@mcp.tool()
def get_price_history(ticker_name: str):
    """
    Return 1 year history of daily Open price, Close price, High price, Low price and trading Volume.
    Ensure the ticker is in normalized format for accurate lookup.
    
    arg: 
        ticker_name: str
    output:
        1 year daily price history: json 
    example:
        {'2023-02-28 00:00:00 EST': {'Open': 17.258894515434886,
                                     'High': 17.371392171233836,
                                     'Low': 17.09014892578125,
                                     'Close': 17.09014892578125,
                                     'Volume': 45100},
         '2023-03-01 00:00:00 EST': {'Open': 17.090151299382544,
                                     'High': 17.094839670907174,
                                     'Low': 16.443295499989794,
                                     'Close': 16.87453269958496,
                                     'Volume': 104300},
         ...
         }
    """
    return kg.get_price_history(ticker_name)

@mcp.tool()
def get_detailed_price_history(ticker_name: str):
    """ 
    Return the past 5 days' history of 1 minute Open price, Close price, High price, Low price and trading Volume, starting from 09:30:00 EST to 15:59:00 EST. Note that the Open, Close, High, Low, Volume are the data for the 1 min duration. However, the Open at 9:30:00 EST may not be equal to the daily Open price, and Close at 15:59:00 EST may not be equal to the daily Close price, due to handling of the paper trade. The sum of the 1 minute Volume may not be equal to the daily Volume.
    Ensure the ticker is in normalized format for accurate lookup.
    
    arg: 
        ticker_name: str
    output:
        past 5 days' 1 min price history: json  
    example:
        {'2024-02-22 09:30:00 EST': {'Open': 15.920000076293945,
                                     'High': 15.920000076293945,
                                     'Low': 15.920000076293945,
                                     'Close': 15.920000076293945,
                                     'Volume': 629},
         '2024-02-22 09:31:00 EST': {'Open': 15.989999771118164,
                                     'High': 15.989999771118164,
                                     'Low': 15.989999771118164,
                                     'Close': 15.989999771118164,
                                     'Volume': 108},
          ...
        }
    """
    return kg.get_detailed_price_history(ticker_name)

@mcp.tool()
def get_dividends_history(ticker_name: str):
    """
    Return dividend history of a ticker.
    Ensure the ticker is in normalized format for accurate lookup.
    
    arg: 
        ticker_name: str
    output:
        dividend distribution history: json
    example:
        {'2019-12-19 00:00:00 EST': 0.058,
         '2020-03-19 00:00:00 EST': 0.2,
         '2020-06-12 00:00:00 EST': 0.2,
         ...
         }
    """
    return kg.get_dividends_history(ticker_name)

@mcp.tool()
def get_market_capitalization(ticker_name: str):
    """
    Return the market capitalization of a ticker.
    Ensure the ticker is in normalized format for accurate lookup.
    
    arg: 
        ticker_name: str
    output:
        market capitalization: float
    """
    return kg.get_market_capitalization(ticker_name)

@mcp.tool()
def get_eps(ticker_name: str):
    """
    Return earnings per share of a ticker.
    Ensure the ticker is in normalized format for accurate lookup.
    
    arg: 
        ticker_name: str
    output:
        earnings per share: float
    """
    return kg.get_eps(ticker_name)

@mcp.tool()
def get_pe_ratio(ticker_name: str):
    """
    Return price-to-earnings ratio of a ticker.
    Ensure the ticker is in normalized format for accurate lookup.
    
    arg: 
        ticker_name: str
    output:
        price-to-earnings ratio: float
    """
    return kg.get_pe_ratio(ticker_name)

@mcp.tool()
def get_info(ticker_name: str):
    """
    Return meta data of a ticker.
    Ensure the ticker is in normalized format for accurate lookup.
    
    arg: 
        ticker_name: str
    output:
        meta information: json
    """
    return kg.get_info(ticker_name)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

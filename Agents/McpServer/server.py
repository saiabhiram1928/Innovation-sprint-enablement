from mcp.server.fastmcp import FastMCP
import sqlite3
import logging

mcp = FastMCP("Demo")

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_db():
    conn = sqlite3.connect("demo.db")
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("Database and table created successfully")

@mcp.prompt(title= "Get_user_prompt")
def get_user_prompt(user_input : str) -> str:
    return f"""Given an user_inout like query mail : mail_id or name : sai from the users table which has schema , id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            Here is the query :  {user_input}
            return the data
            """

@mcp.tool()
def add_data(query: str) -> str:  # Changed return type to str for better error reporting
    """Insert data to the table"""
    logger.info(f"Attempting to execute query: {query}")
    try:
        conn = sqlite3.connect("demo.db")
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        success_msg = f"Successfully inserted {rows_affected} row(s)"
        logger.info(success_msg)
        return success_msg
        
    except sqlite3.IntegrityError as e:
        error_msg = f"Database integrity error: {e} (possibly duplicate email)"
        logger.error(error_msg)
        return error_msg
    except sqlite3.Error as e:
        error_msg = f"Database error: {e}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
def query_data(query: str) -> str:  # Changed return type to str for better formatting
    """Queries data if not found will return all the data"""
    logger.info(f"Attempting to execute query: {query}")
    try:
        conn = sqlite3.connect("demo.db")
        cursor = conn.cursor()
        results = cursor.execute(query).fetchall()
        conn.close()
        
        if not results:
            return "No data found"
        
        # Format results nicely
        formatted_results = []
        for row in results:
            formatted_results.append(str(row))
        
        result_str = f"Found {len(results)} row(s):\n" + "\n".join(formatted_results)
        logger.info(f"Query returned {len(results)} rows")
        return result_str
        
    except sqlite3.Error as e:
        error_msg = f"Database error: {e}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
def list_all_users() -> str:
    """List all users in the database for debugging"""
    return query_data("SELECT * FROM users")

@mcp.tool()
def check_table_structure() -> str:
    """Check the table structure for debugging"""
    return query_data("PRAGMA table_info(users)")

create_db()

if __name__ == "__main__":
    mcp.run()








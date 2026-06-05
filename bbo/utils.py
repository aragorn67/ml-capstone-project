"""
bbo/utils.py — Formatting and display utilities
"""


def format_portal_string(query):
    """Convert a query array to the portal submission format."""
    return "-".join([f"{v:.6f}" for v in query])


def print_function_header(func_num, n_dim, n_points, strategy,
                          best_output, best_input):
    """Print the header for a function's query recommendation."""
    print(f"{'='*65}")
    print(f"F{func_num} ({n_dim}D) | {n_points} pts | {strategy} | "
          f"best={best_output:.4e}")
    print(f"  best input: {best_input}")
    print(f"{'='*65}")


def print_summary(portal_strings):
    """Print the portal submission summary."""
    print("=" * 65)
    print("  PORTAL SUBMISSION SUMMARY")
    print("=" * 65)
    for i in range(1, 9):
        print(f"  Function {i}: {portal_strings[i]}")
    print("=" * 65)

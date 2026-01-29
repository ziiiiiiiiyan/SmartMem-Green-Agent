def print_table(headers, rows):
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Create format string
    fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])
    separator = "-+-".join(["-" * w for w in col_widths])
    
    output = []
    output.append(fmt.format(*headers))
    output.append(separator)
    for row in rows:
        output.append(fmt.format(*[str(c) for c in row]))
    return "\n".join(output)
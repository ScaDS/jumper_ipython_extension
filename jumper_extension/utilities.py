import pandas as pd


def filter_perfdata(cell_history_data, perfdata, compress_idle=True):
    """Filter performance data to remove idle periods if requested"""
    if cell_history_data is None or cell_history_data.empty:
        return perfdata.iloc[0:0]

    if compress_idle:
        # Remove idle periods between cells
        # Create time masks for each cell's execution period
        masks = []
        for _, cell in cell_history_data.iterrows():
            mask = (perfdata["time"] >= cell["start_time"]) & (
                perfdata["time"] <= cell["end_time"]
            )
            masks.append(mask)

        if masks:
            combined_mask = pd.concat(masks, axis=1).any(axis=1)
            return perfdata[combined_mask]
        else:
            return perfdata.iloc[0:0]
    else:
        # Get start time from first cell and end time from last cell in the range
        start_time = cell_history_data.iloc[0]["start_time"]
        end_time = cell_history_data.iloc[-1]["end_time"]
        return perfdata[
            (perfdata["time"] >= start_time) & (perfdata["time"] <= end_time)
        ]

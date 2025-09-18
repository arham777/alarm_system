# Unhealthy Sources Chart - Data Flow Explanation

## Current Architecture:

### Overall Health Endpoint (Working):
```
Frontend → /pvcI-health/overall → Pre-saved JSON file → Fast Response
```

### Unhealthy Sources Endpoint (Not Working):
```
Frontend → /pvcI-health/unhealthy-sources → Real-time CSV processing → Slow/Empty Response
```

## What the Unhealthy Sources Chart is Trying to Show:

### 1. **Timeline View:**
- **X-axis**: Time (when unhealthy period occurred)
- **Y-axis**: Source names (alarm sources like sensors, equipment)
- **Dot size**: Number of alarms in 10-minute window
- **Dot color**: Priority (Red=High, Yellow=Medium, Green=Low)

### 2. **Bar Chart View:**
- Shows top sources by total alarm hits
- Helps identify most problematic sources

### 3. **Real Example:**
```json
{
  "count": 3,
  "records": [
    {
      "event_time": "2025-09-18T10:00:00Z",
      "source": "PVC1-Temperature-Sensor-01",
      "hits": 15,
      "threshold": 10,
      "over_by": 5,
      "priority": "High",
      "location_tag": "Reactor Area",
      "condition": "High Temperature"
    },
    {
      "event_time": "2025-09-18T10:10:00Z", 
      "source": "PVC1-Pressure-Valve-03",
      "hits": 12,
      "threshold": 10,
      "over_by": 2,
      "priority": "Medium",
      "location_tag": "Control Room",
      "condition": "Pressure Deviation"
    }
  ]
}
```

## Why No Data is Coming:

1. **Real CSV files might not have recent unhealthy data**
2. **CSV processing is taking too long and timing out**
3. **Time zone issues between frontend and backend**
4. **Threshold too high (10 alarms per 10 minutes)**

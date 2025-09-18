// Test different time ranges for unhealthy sources
const API_BASE_URL = 'http://localhost:8000';

async function testTimeRanges() {
    console.log('Testing different time ranges for unhealthy sources...\n');
    
    // Test 1: Current 24h (likely empty)
    console.log('=== Test 1: Current 24h ===');
    const endTime = new Date();
    const startTime = new Date();
    startTime.setDate(endTime.getDate() - 1);
    
    try {
        const url1 = `${API_BASE_URL}/pvcI-health/unhealthy-sources?bin_size=10T&alarm_threshold=10&start_time=${startTime.toISOString()}&end_time=${endTime.toISOString()}`;
        console.log('URL:', url1);
        
        const response1 = await fetch(url1);
        if (response1.ok) {
            const data1 = await response1.json();
            console.log(`Result: ${data1.count} sources found`);
            if (data1.count > 0) {
                console.log('Sample source:', data1.records[0].source);
            }
        }
    } catch (error) {
        console.error('Error:', error.message);
    }
    
    // Test 2: No time filter (all historical data)
    console.log('\n=== Test 2: No Time Filter (All Historical) ===');
    try {
        const url2 = `${API_BASE_URL}/pvcI-health/unhealthy-sources?bin_size=10T&alarm_threshold=10`;
        console.log('URL:', url2);
        
        const response2 = await fetch(url2);
        if (response2.ok) {
            const data2 = await response2.json();
            console.log(`Result: ${data2.count} sources found`);
            if (data2.count > 0) {
                console.log('Sample sources:');
                data2.records.slice(0, 5).forEach((record, index) => {
                    console.log(`${index + 1}. ${record.source} - ${record.hits} hits at ${record.event_time}`);
                });
            }
        }
    } catch (error) {
        console.error('Error:', error.message);
    }
    
    // Test 3: Specific historical date (March 2025)
    console.log('\n=== Test 3: Specific Historical Date (March 2025) ===');
    try {
        const historicalStart = '2025-03-14T15:00:00Z';
        const historicalEnd = '2025-03-14T16:00:00Z';
        
        const url3 = `${API_BASE_URL}/pvcI-health/unhealthy-sources?bin_size=10T&alarm_threshold=10&start_time=${historicalStart}&end_time=${historicalEnd}`;
        console.log('URL:', url3);
        
        const response3 = await fetch(url3);
        if (response3.ok) {
            const data3 = await response3.json();
            console.log(`Result: ${data3.count} sources found`);
            if (data3.count > 0) {
                console.log('Historical sources:');
                data3.records.slice(0, 5).forEach((record, index) => {
                    console.log(`${index + 1}. ${record.source} - ${record.hits} hits (${record.condition})`);
                });
            }
        }
    } catch (error) {
        console.error('Error:', error.message);
    }
}

testTimeRanges();

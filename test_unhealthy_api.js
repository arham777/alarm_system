// Simple test script to check the unhealthy sources API
const API_BASE_URL = 'http://localhost:8000';

async function testUnhealthySourcesAPI() {
    try {
        console.log('Testing unhealthy sources API...');
        
        // Test with different time ranges
        const timeRanges = [
            { name: '1 hour', hours: 1 },
            { name: '6 hours', hours: 6 },
            { name: '24 hours', hours: 24 },
            { name: '7 days', hours: 24 * 7 }
        ];
        
        for (const range of timeRanges) {
            const endTime = new Date();
            const startTime = new Date();
            startTime.setHours(endTime.getHours() - range.hours);
            
            const url = new URL(`${API_BASE_URL}/pvcI-health/unhealthy-sources`);
            url.searchParams.set('bin_size', '10T');
            url.searchParams.set('alarm_threshold', '10');
            url.searchParams.set('start_time', startTime.toISOString());
            url.searchParams.set('end_time', endTime.toISOString());
            
            console.log(`\n--- Testing ${range.name} range ---`);
            console.log(`URL: ${url.toString()}`);
            
            try {
                const response = await fetch(url.toString());
                
                if (!response.ok) {
                    console.error(`HTTP Error: ${response.status} ${response.statusText}`);
                    continue;
                }
                
                const data = await response.json();
                console.log(`Response:`, {
                    count: data.count,
                    recordsLength: data.records?.length || 0,
                    sampleRecord: data.records?.[0] || null
                });
                
                if (data.count > 0) {
                    console.log(`✅ Found ${data.count} unhealthy sources in ${range.name}`);
                    break; // Found data, no need to test other ranges
                } else {
                    console.log(`❌ No unhealthy sources found in ${range.name}`);
                }
                
            } catch (fetchError) {
                console.error(`Fetch error for ${range.name}:`, fetchError.message);
            }
        }
        
        // Test without time filters
        console.log('\n--- Testing without time filters ---');
        const noFilterUrl = `${API_BASE_URL}/pvcI-health/unhealthy-sources?bin_size=10T&alarm_threshold=10`;
        console.log(`URL: ${noFilterUrl}`);
        
        try {
            const response = await fetch(noFilterUrl);
            if (response.ok) {
                const data = await response.json();
                console.log(`No filter response:`, {
                    count: data.count,
                    recordsLength: data.records?.length || 0,
                    sampleRecord: data.records?.[0] || null
                });
            } else {
                console.error(`HTTP Error: ${response.status} ${response.statusText}`);
            }
        } catch (error) {
            console.error('No filter fetch error:', error.message);
        }
        
        // Test overall health to see if there's any data at all
        console.log('\n--- Testing overall health endpoint ---');
        try {
            const healthResponse = await fetch(`${API_BASE_URL}/pvcI-health/overall`);
            if (healthResponse.ok) {
                const healthData = await healthResponse.json();
                console.log('Overall health data:', {
                    totalSources: healthData.overall?.totals?.sources || 0,
                    totalFiles: healthData.overall?.totals?.files || 0,
                    healthPct: healthData.overall?.health_pct_simple || 0,
                    unhealthyBins: Object.keys(healthData.overall?.unhealthy_sources_by_bins || {}).length
                });
            }
        } catch (error) {
            console.error('Health endpoint error:', error.message);
        }
        
    } catch (error) {
        console.error('Test failed:', error);
    }
}

// Run the test
testUnhealthySourcesAPI();

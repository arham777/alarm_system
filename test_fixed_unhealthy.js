// Test the fixed unhealthy sources API
const API_BASE_URL = 'http://localhost:8000';

async function testFixedUnhealthyAPI() {
    try {
        console.log('Testing fixed unhealthy sources approach...');
        
        // First check overall health to see what data is available
        console.log('\n1. Checking overall health data...');
        const healthResponse = await fetch(`${API_BASE_URL}/pvcI-health/overall`);
        
        if (!healthResponse.ok) {
            console.error('Overall health API failed:', healthResponse.status);
            return;
        }
        
        const healthData = await healthResponse.json();
        
        console.log('Overall Health Summary:');
        console.log('- Health %:', healthData.overall?.health_pct_simple || 0);
        console.log('- Total Sources:', healthData.overall?.totals?.sources || 0);
        console.log('- Total Files:', healthData.overall?.totals?.files || 0);
        
        // Check unhealthy sources by bins
        const unhealthyBins = healthData.overall?.unhealthy_sources_by_bins || {};
        console.log('\n2. Unhealthy Sources by Bins:');
        
        if (Object.keys(unhealthyBins).length === 0) {
            console.log('❌ No unhealthy sources found in bins');
        } else {
            let totalUnhealthySources = 0;
            Object.entries(unhealthyBins).forEach(([binRange, sources]) => {
                console.log(`\n📊 Bin Range ${binRange}: ${sources.length} sources`);
                totalUnhealthySources += sources.length;
                
                // Show first few sources
                sources.slice(0, 2).forEach((source, index) => {
                    console.log(`   ${index + 1}. ${source.filename} - ${source.unhealthy_bins} unhealthy bins (${source.health_pct}% healthy)`);
                });
            });
            
            console.log(`\n✅ Total unhealthy sources found: ${totalUnhealthySources}`);
        }
        
        // Test the unhealthy sources endpoint (should now work with fallback)
        console.log('\n3. Testing unhealthy sources endpoint with fallback...');
        
        try {
            const unhealthyResponse = await fetch(`${API_BASE_URL}/pvcI-health/unhealthy-sources?bin_size=10T&alarm_threshold=10`);
            
            if (unhealthyResponse.ok) {
                const unhealthyData = await unhealthyResponse.json();
                console.log('Unhealthy Sources Response:');
                console.log('- Count:', unhealthyData.count);
                console.log('- Records Length:', unhealthyData.records?.length || 0);
                
                if (unhealthyData.count > 0) {
                    console.log('\n✅ Sample unhealthy source record:');
                    const sample = unhealthyData.records[0];
                    console.log({
                        source: sample.source,
                        hits: sample.hits,
                        priority: sample.priority,
                        location: sample.location_tag,
                        condition: sample.condition
                    });
                } else {
                    console.log('❌ Still no unhealthy sources in API response');
                }
            } else {
                console.error('Unhealthy sources API failed:', unhealthyResponse.status);
            }
        } catch (apiError) {
            console.error('API call failed:', apiError.message);
        }
        
    } catch (error) {
        console.error('Test failed:', error.message);
    }
}

testFixedUnhealthyAPI();

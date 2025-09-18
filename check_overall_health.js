// Check overall health API to see if it has unhealthy data
const API_BASE_URL = 'http://localhost:8000';

async function checkOverallHealthData() {
    try {
        console.log('Checking overall health API for unhealthy data...');
        
        const response = await fetch(`${API_BASE_URL}/pvcI-health/overall`);
        
        if (!response.ok) {
            console.error(`HTTP Error: ${response.status}`);
            return;
        }
        
        const data = await response.json();
        
        console.log('\n=== Overall Health Data ===');
        console.log('Health Percentage:', data.overall?.health_pct_simple || 0);
        console.log('Unhealthy Percentage:', data.overall?.unhealthy_percentage || 0);
        console.log('Total Sources:', data.overall?.totals?.sources || 0);
        console.log('Total Files:', data.overall?.totals?.files || 0);
        
        // Check unhealthy sources by bins
        const unhealthyBins = data.overall?.unhealthy_sources_by_bins || {};
        console.log('\n=== Unhealthy Sources by Bins ===');
        
        if (Object.keys(unhealthyBins).length === 0) {
            console.log('❌ No unhealthy sources found in any bins');
        } else {
            Object.entries(unhealthyBins).forEach(([binRange, sources]) => {
                console.log(`\n📊 Bin Range: ${binRange}`);
                console.log(`   Sources Count: ${sources.length}`);
                
                sources.slice(0, 3).forEach((source, index) => {
                    console.log(`   ${index + 1}. ${source.filename} - ${source.unhealthy_bins} unhealthy bins`);
                });
                
                if (sources.length > 3) {
                    console.log(`   ... and ${sources.length - 3} more sources`);
                }
            });
        }
        
        // Check if we have files data
        if (data.files && data.files.length > 0) {
            console.log('\n=== Files Data Sample ===');
            const unhealthyFiles = data.files.filter(f => f.unhealthy_bins > 0);
            console.log(`Unhealthy files found: ${unhealthyFiles.length}`);
            
            unhealthyFiles.slice(0, 5).forEach((file, index) => {
                console.log(`${index + 1}. ${file.filename} - ${file.unhealthy_bins} unhealthy bins (${file.health_pct}% healthy)`);
            });
        }
        
    } catch (error) {
        console.error('Error checking overall health:', error.message);
    }
}

checkOverallHealthData();

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

async function main() {
    const allowedDirs = [
        'C:\\Users\\4artu\\soft\\antigravity-remove',
        // 'C:\\tmp',
        // process.cwd()
    ];

    const transport = new StdioClientTransport({
        command: 'npx',
        args: ['@modelcontextprotocol/server-filesystem', ...allowedDirs]
    });

    const client = new Client(
        { name: 'multi-fs-client', version: '1.0.0' },
        { capabilities: {} }
    );

    try {
        await client.connect(transport);

        const allowedResult = await client.callTool({
            name: 'list_allowed_directories',
            arguments: {}
        });

        console.log('Allowed directories:');
        console.log(allowedResult.content[0].text);

        for (const dir of allowedDirs) {
            try {
                console.log(`\n=== ${dir} ===`);

                const listResult = await client.callTool({
                    name: 'list_directory',
                    arguments: { path: dir }
                });

                const content = listResult.content[0].text;
                const lines = content.split('\n').filter(l => l.trim());

                console.log(`Items: ${lines.length}`);

                if (lines.length > 0) {
                    console.log('First few items:');
                    lines.slice(0, 5).forEach(line => console.log(`  ${line}`));
                }

            } catch (error) {
                console.log(`Skipped: ${error.message}`);
            }
        }

    } catch (error) {
        console.error('Error:', error.message);
    } finally {
        await client.close();
    }
}

    main().catch(console.error);

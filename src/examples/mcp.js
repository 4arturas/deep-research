import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

async function main() {
    const allowedDirs = [
        'C:\\Users\\4artu\\soft\\antigravity-remove'
    ];

    const transport = new StdioClientTransport({
        command: 'npx',
        args: ['@modelcontextprotocol/server-filesystem', ...allowedDirs]
    });

    const client = new Client(
        { name: 'file-reader', version: '1.0.0' },
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

                console.log(`Total items: ${lines.length}`);

                const files = lines
                    .filter(line => line.includes('[FILE]'))
                    .map(line => line.replace('[FILE] ', '').trim());

                console.log(`Files found: ${files.length}`);

                for (const fileName of files) {
                    const filePath = `${dir}\\${fileName}`;

                    try {
                        console.log(`\n--- Reading: ${fileName} ---`);

                        const fileResult = await client.callTool({
                            name: 'read_text_file',
                            arguments: {
                                path: filePath
                            }
                        });

                        const fileContent = fileResult.content[0].text;
                        console.log(`Content (${fileContent.length} chars):`);
                        console.log(fileContent);

                    } catch (fileError) {
                        console.log(`Failed to read ${fileName}: ${fileError.message}`);
                    }
                }

                try {
                    const getFileInfoResult = await client.callTool({
                        name: 'get_file_info',
                        arguments: {
                            path: dir
                        }
                    });

                    console.log('\nDirectory info:');
                    console.log(getFileInfoResult.content[0].text);

                } catch (infoError) {
                    console.log(`Cannot get directory info: ${infoError.message}`);
                }

                try {
                    const searchResult = await client.callTool({
                        name: 'search_files',
                        arguments: {
                            path: dir,
                            pattern: '*'
                        }
                    });

                    console.log('\n=== Search results ===');
                    console.log(searchResult.content[0].text);
                } catch (searchError) {
                    console.log(`Search error: ${searchError.message}`);
                }

                try {
                    const treeResult = await client.callTool({
                        name: 'directory_tree',
                        arguments: {
                            path: dir
                        }
                    });

                    console.log('\n=== Directory tree ===');
                    console.log(treeResult.content[0].text);
                } catch (treeError) {
                    console.log(`Tree error: ${treeError.message}`);
                }

                try {
                    const listSizesResult = await client.callTool({
                        name: 'list_directory_with_sizes',
                        arguments: {
                            path: dir
                        }
                    });

                    console.log('\n=== Directory with sizes ===');
                    console.log(listSizesResult.content[0].text);
                } catch (sizesError) {
                    console.log(`Sizes error: ${sizesError.message}`);
                }

                try {
                    const headResult = await client.callTool({
                        name: 'read_text_file',
                        arguments: {
                            path: `${dir}\\${files[0]}`,
                            head: 3
                        }
                    });

                    console.log('\n=== First 3 lines ===');
                    console.log(headResult.content[0].text);
                } catch (headError) {
                    console.log(`Head error: ${headError.message}`);
                }

                try {
                    const tailResult = await client.callTool({
                        name: 'read_text_file',
                        arguments: {
                            path: `${dir}\\${files[0]}`,
                            tail: 2
                        }
                    });

                    console.log('\n=== Last 2 lines ===');
                    console.log(tailResult.content[0].text);
                } catch (tailError) {
                    console.log(`Tail error: ${tailError.message}`);
                }

            } catch (error) {
                console.log(`Error processing ${dir}: ${error.message}`);
            }
        }

    } catch (error) {
        console.error('Main error:', error.message);
    } finally {
        await client.close();
    }
}

main().catch(console.error);
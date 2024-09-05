
// Internal state.
var CURRENT_INPUT_GRID = new Grid(3, 3);
var CURRENT_OUTPUT_GRID = new Grid(3, 3);
var TEST_PAIRS = new Array();
var TEST_PAIRS_PRED = new Array();
var TEST_PAIRS_QSTATE = new Array();
var CURRENT_TEST_PAIR_INDEX = 0;
var COPY_PASTE_DATA = new Array();
var pred_array = new Array();
var task;
var currentIndex;
var train;
var test;


// Cosmetic.
var EDITION_GRID_HEIGHT = 500;
var EDITION_GRID_WIDTH = 500;
var MAX_CELL_SIZE = 100;


function resetTask() {
    CURRENT_INPUT_GRID = new Grid(3, 3);
    TEST_PAIRS = new Array();
    TEST_PAIRS_PRED = new Array();
    TEST_PAIRS_QSTATE = new Array();
    CURRENT_TEST_PAIR_INDEX = 0;
    $('#task_preview').html('');
    resetOutputGrid();
}

function refreshEditionGrid(jqGrid, dataGrid) {
    fillJqGridWithData(jqGrid, dataGrid);
    setUpEditionGridListeners(jqGrid);
    fitCellsToContainer(jqGrid, dataGrid.height, dataGrid.width, EDITION_GRID_HEIGHT, EDITION_GRID_HEIGHT);
    initializeSelectable();
}

function syncFromEditionGridToDataGrid() {
    copyJqGridToDataGrid($('#output_grid .edition_grid'), CURRENT_OUTPUT_GRID);
}

function syncFromDataGridToEditionGrid() {
    refreshEditionGrid($('#output_grid .edition_grid'), CURRENT_OUTPUT_GRID);
}

function getSelectedSymbol() {
    selected = $('#symbol_picker .selected-symbol-preview')[0];
    return $(selected).attr('symbol');
}

function setUpEditionGridListeners(jqGrid) {
    jqGrid.find('.cell').click(function(event) {
        cell = $(event.target);
        symbol = getSelectedSymbol();

        mode = $('input[name=tool_switching]:checked').val();
        if (mode == 'floodfill') {
            // If floodfill: fill all connected cells.
            syncFromEditionGridToDataGrid();
            grid = CURRENT_OUTPUT_GRID.grid;
            floodfillFromLocation(grid, cell.attr('x'), cell.attr('y'), symbol);
            syncFromDataGridToEditionGrid();
        }
        else if (mode == 'edit') {
            // Else: fill just this cell.
            setCellSymbol(cell, symbol);
        }
    });
}

function resizeOutputGrid() {
    size = $('#output_grid_size').val();
    size = parseSizeTuple(size);
    height = size[0];
    width = size[1];

    jqGrid = $('#output_grid .edition_grid');
    syncFromEditionGridToDataGrid();
    dataGrid = JSON.parse(JSON.stringify(CURRENT_OUTPUT_GRID.grid));
    CURRENT_OUTPUT_GRID = new Grid(height, width, dataGrid);
    refreshEditionGrid(jqGrid, CURRENT_OUTPUT_GRID);
}

function resetOutputGrid() {
    syncFromEditionGridToDataGrid();
    CURRENT_OUTPUT_GRID = new Grid(3, 3);
    syncFromDataGridToEditionGrid();
    resizeOutputGrid();
}

function copyFromInput() {
    syncFromEditionGridToDataGrid();
    CURRENT_OUTPUT_GRID = convertSerializedGridToGridObject(CURRENT_INPUT_GRID.grid);
    syncFromDataGridToEditionGrid();
    $('#output_grid_size').val(CURRENT_OUTPUT_GRID.height + 'x' + CURRENT_OUTPUT_GRID.width);
}

function fillPairPreview(pairId, inputGrid, outputGrid) {
    var pairSlot = $('#pair_preview_' + pairId);
    if (!pairSlot.length) {
        // Create HTML for pair.
        pairSlot = $('<div id="pair_preview_' + pairId + '" class="pair_preview" index="' + pairId + '"></div>');
        pairSlot.appendTo('#task_preview');
    }
    var jqInputGrid = pairSlot.find('.input_preview');
    if (!jqInputGrid.length) {
        jqInputGrid = $('<div class="input_preview"></div>');
        jqInputGrid.appendTo(pairSlot);
    }
    var jqOutputGrid = pairSlot.find('.output_preview');
    if (!jqOutputGrid.length) {
        jqOutputGrid = $('<div class="output_preview"></div>');
        jqOutputGrid.appendTo(pairSlot);
    }

    fillJqGridWithData(jqInputGrid, inputGrid);
    fitCellsToContainer(jqInputGrid, inputGrid.height, inputGrid.width, 200, 200);
    fillJqGridWithData(jqOutputGrid, outputGrid);
    fitCellsToContainer(jqOutputGrid, outputGrid.height, outputGrid.width, 200, 200);
}

function loadJSONTask(train, test) {
    resetTask();
    $('#modal_bg').hide();
    $('#error_display').hide();
    $('#info_display').hide();

    for (var i = 0; i < train.length; i++) {
        pair = train[i];
        values = pair['input'];
        input_grid = convertSerializedGridToGridObject(values)
        values = pair['output'];
        output_grid = convertSerializedGridToGridObject(values)
        fillPairPreview(i, input_grid, output_grid);
    }
    for (var i=0; i < test.length; i++) {
        pair = test[i];
        TEST_PAIRS.push(pair);
        // console.log(typeof pair);
        
    }

    // for (var i=0; i < test.length; i++) {
    //     pair = test[i];
    //     pair.pred = pred_array[i][4][0];  // important
    //     TEST_PAIRS_PRED.push(pair);
    //     // console.log(typeof pair);
        
    // }

    // console.log(TEST_PAIRS)
    // console.log(TEST_PAIRS_PRED)

   
    values = TEST_PAIRS[0]['input'];
    CURRENT_INPUT_GRID = convertSerializedGridToGridObject(values)
    fillTestInput(CURRENT_INPUT_GRID);


    values_o = TEST_PAIRS[0]['output'];
    CURRENT_OUTPUT_GRID = convertSerializedGridToGridObject(values_o)
    fillTestOutput(CURRENT_OUTPUT_GRID);

    
    // values_pred = TEST_PAIRS_PRED[0]['pred'];
    // CURRENT_OUTPUT_PRED_GRID = convertSerializedGridToGridObject(values_pred)
    // fillTestOutputPred(CURRENT_OUTPUT_PRED_GRID);




    CURRENT_TEST_PAIR_INDEX = 0;
    $('#current_test_input_id_display').html('1');
    $('#total_test_input_count_display').html(test.length);
}

function display_task_name(task_name, task_index, number_of_tasks) {
    big_space = '&nbsp;'.repeat(4); 
    document.getElementById('task_name').innerHTML = (
        'Task name:' + big_space + task_name + big_space + (
            task_index===null ? '' :
            ( String(task_index) + ' out of ' + String(number_of_tasks) )
        )
    );
}

function loadTaskFromFile(e) {
    currentIndex = 0;
    updateIndexValue();
    document.getElementById('evaluation_predicted_output').innerHTML = '';
    document.getElementById('evaluation_qstate_output').innerHTML = '';
    var file = e.target.files[0];
    if (!file) {
        errorMsg('No file selected');
        return;
    }
    var reader = new FileReader();
    reader.onload = function(e) {
        var contents = e.target.result;

        try {
            contents = JSON.parse(contents);
            task = { "name": file.name, "contents": contents }; // Update global task variable
            train = contents['train'];
            test = contents['test'];
        } catch (e) {
            errorMsg('Bad file format');
            return;
        }
        loadJSONTask(train, test);

        $('#load_task_file_input')[0].value = "";
        infoMsg("Loaded task training/" + task["name"]);
        display_task_name(task['name']);
    };
    reader.readAsText(file);
}






function randomTask() {
    currentIndex = 0;
    updateIndexValue();
    document.getElementById('evaluation_predicted_output').innerHTML = '';
    document.getElementById('evaluation_qstate_output').innerHTML = '';
    
    var subset = "training";
    // Fetch the list of tasks from the GitHub API

    $.getJSON("https://api.github.com/repos/fchollet/ARC/contents/data/" + subset, function(tasks) {
        // Select a random task from the list
        var task_index = Math.floor(Math.random() * tasks.length);
        task = tasks[task_index];  // Set the global task variable
        // Fetch the task details from GitHub and load the task
        $.getJSON(task["download_url"], function(json) {
            try {
                train = json['train'];
                test = json['test'];
            } catch (e) {
                errorMsg('Bad file format');
                return;
            }
            loadJSONTask(train, test);
            infoMsg("Loaded task training/" + task["name"]);
            display_task_name(task['name'], task_index, tasks.length);
        }).fail(function() {
            errorMsg('Error loading task');
        });
    }).fail(function() {
        errorMsg('Error loading task list');
    });
}

function runAgent() {
    if (!task || !task['name']) {
        errorMsg('No task selected. Please load a task first.');
        return;
    }

    // Display the status message
    document.getElementById('agentStatus').innerText = 'AGENT IS RUNNING...';

    $.ajax({
        url: 'http://127.0.0.1:5000/process_task',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ task_name: task['name'] }),
        success: function(response) {
            try {
                pred_array = JSON.parse(response.pred_array);
                console.log(pred_array, 'Hi');
                console.log(pred_array.length, 'Hi 2');
                console.log(typeof pred_array, 'Hi 2');

                // Clear the status message
                document.getElementById('agentStatus').innerText = '';

                updateIndexValue();
                updateOutput();
            } catch (e) {
                errorMsg('Error processing response from server');
                console.error(e);
            }
        },
        error: function() {
            errorMsg('Error processing task on server');
            // Clear the status message
            document.getElementById('agentStatus').innerText = '';
        }
    });
}









currentIndex = 0; // Initial index value

function updateIndexValue() {
    document.getElementById("indexValue").innerText = currentIndex;
}

function increaseIndex() {
    currentIndex++;
    updateIndexValue();
    updateOutput();
}

function decreaseIndex() {
    if (currentIndex > 0) { // Ensure the index doesn't go below 0
        currentIndex--;
    }
    updateIndexValue();
    updateOutput();
}

function updateOutput() {
    var selectedIndex = currentIndex;
    TEST_PAIRS_QSTATE = [];
    TEST_PAIRS_PRED = [];

    for (var i = 0; i < pred_array.length; i++) {
        var pair = {};
        pair.qstate = pred_array[i][selectedIndex][1];  // important
        TEST_PAIRS_QSTATE.push(pair);
    }

    for (var i = 0; i < pred_array.length; i++) {
        var pair = {};
        pair.pred = pred_array[i][selectedIndex][0];  // important
        TEST_PAIRS_PRED.push(pair);
    }

    if (TEST_PAIRS_QSTATE.length > 0) {
        var values_qstate = TEST_PAIRS_QSTATE[0]['qstate'];
        var CURRENT_OUTPUT_QSTATE_GRID = convertSerializedGridToGridObject(values_qstate);
        fillTestOutputQstate(CURRENT_OUTPUT_QSTATE_GRID);

        var values_pred = TEST_PAIRS_PRED[0]['pred'];
        CURRENT_OUTPUT_PRED_GRID = convertSerializedGridToGridObject(values_pred);
        fillTestOutputPred(CURRENT_OUTPUT_PRED_GRID);
    } else {
        console.log("No valid qstate found for the selected index.");
    }
}












function nextTestInput() {
    if (TEST_PAIRS.length <= CURRENT_TEST_PAIR_INDEX + 1) {
        errorMsg('No next test input. Pick another file?')
        return
    }
    CURRENT_TEST_PAIR_INDEX += 1;
    values = TEST_PAIRS[CURRENT_TEST_PAIR_INDEX]['input'];
    values_o = TEST_PAIRS[CURRENT_TEST_PAIR_INDEX]['output'];
    values_pred = TEST_PAIRS_PRED[CURRENT_TEST_PAIR_INDEX]['pred'];
    values_qstate = TEST_PAIRS_QSTATE[CURRENT_TEST_PAIR_INDEX]['qstate'];
    CURRENT_INPUT_GRID = convertSerializedGridToGridObject(values)
    CURRENT_OUTPUT_GRID = convertSerializedGridToGridObject(values_o)
    CURRENT_OUTPUT_PRED_GRID = convertSerializedGridToGridObject(values_pred)
    CURRENT_OUTPUT_QSTATE_GRID = convertSerializedGridToGridObject(values_qstate)
    fillTestInput(CURRENT_INPUT_GRID);
    fillTestOutput(CURRENT_OUTPUT_GRID);
    fillTestOutputPred(CURRENT_OUTPUT_PRED_GRID);
    fillTestOutputQstate(CURRENT_OUTPUT_QSTATE_GRID);

    $('#current_test_input_id_display').html(CURRENT_TEST_PAIR_INDEX + 1);
    $('#total_test_input_count_display').html(test.length);
}


function previousTestInput() {
    if (CURRENT_TEST_PAIR_INDEX <= 0) {
        errorMsg('No previous test input. Pick another file?');
        return;
    }
    // if (TEST_PAIRS.length >= CURRENT_TEST_PAIR_INDEX + 1) {
    //     errorMsg('No previous test input. Pick another file?')
    //     return
    // }
    CURRENT_TEST_PAIR_INDEX -= 1;
    values = TEST_PAIRS[CURRENT_TEST_PAIR_INDEX]['input'];
    values_o = TEST_PAIRS[CURRENT_TEST_PAIR_INDEX]['output'];
    values_pred = TEST_PAIRS_PRED[CURRENT_TEST_PAIR_INDEX]['pred'];
    values_qstate = TEST_PAIRS_QSTATE[CURRENT_TEST_PAIR_INDEX]['qstate'];
    CURRENT_INPUT_GRID = convertSerializedGridToGridObject(values);
    CURRENT_OUTPUT_GRID = convertSerializedGridToGridObject(values_o);
    CURRENT_OUTPUT_PRED_GRID = convertSerializedGridToGridObject(values_pred);
    CURRENT_OUTPUT_QSTATE_GRID = convertSerializedGridToGridObject(values_qstate);
    fillTestInput(CURRENT_INPUT_GRID);
    fillTestOutput(CURRENT_OUTPUT_GRID);
    fillTestOutputPred(CURRENT_OUTPUT_PRED_GRID);
    fillTestOutputQstate(CURRENT_OUTPUT_QSTATE_GRID);

    $('#current_test_input_id_display').html(CURRENT_TEST_PAIR_INDEX + 1);
    $('#total_test_input_count_display').html(test.length);
}




function submitSolution() {
    syncFromEditionGridToDataGrid();
    reference_output = TEST_PAIRS[CURRENT_TEST_PAIR_INDEX]['output'];
    submitted_output = CURRENT_OUTPUT_PRED_GRID.grid;
    if (reference_output.length != submitted_output.length) {
        errorMsg('Wrong solution.');
        return
    }
    for (var i = 0; i < reference_output.length; i++){
        ref_row = reference_output[i];
        for (var j = 0; j < ref_row.length; j++){
            if (ref_row[j] != submitted_output[i][j]) {
                errorMsg('Wrong solution.');
                return
            }
        }

    }
    infoMsg('Correct solution!');
}

function fillTestInput(inputGrid) {
    jqInputGrid = $('#evaluation_input');
    fillJqGridWithData(jqInputGrid, inputGrid);
    fitCellsToContainer(jqInputGrid, inputGrid.height, inputGrid.width, 400, 400);
}

function fillTestOutput(outputGrid) {
    jqOutputGrid = $('#evaluation_output');
    fillJqGridWithData(jqOutputGrid, outputGrid);
    fitCellsToContainer(jqOutputGrid, outputGrid.height, outputGrid.width, 400, 400);
}

function fillTestOutputPred(outputPredGrid) {
    jqOutputPredGrid = $('#evaluation_predicted_output');
    fillJqGridWithData(jqOutputPredGrid, outputPredGrid);
    fitCellsToContainer(jqOutputPredGrid, outputPredGrid.height, outputPredGrid.width, 400, 400);
    
}

function fillTestOutputQstate(outputQstateGrid) {
    jqOutputQstateGrid = $('#evaluation_qstate_output');
    fillJqGridWithData(jqOutputQstateGrid, outputQstateGrid);
    fitCellsToContainer(jqOutputQstateGrid, outputQstateGrid.height, outputQstateGrid.width, 400, 400);
    
}

function copyToOutput() {
    syncFromEditionGridToDataGrid();
    CURRENT_OUTPUT_GRID = convertSerializedGridToGridObject(CURRENT_INPUT_GRID.grid);
    syncFromDataGridToEditionGrid();
    $('#output_grid_size').val(CURRENT_OUTPUT_GRID.height + 'x' + CURRENT_OUTPUT_GRID.width);
}

function initializeSelectable() {
    try {
        $('.selectable_grid').selectable('destroy');
    }
    catch (e) {
    }
    toolMode = $('input[name=tool_switching]:checked').val();
    if (toolMode == 'select') {
        infoMsg('Select some cells and click on a color to fill in, or press C to copy');
        $('.selectable_grid').selectable(
            {
                autoRefresh: false,
                filter: '> .row > .cell',
                start: function(event, ui) {
                    $('.ui-selected').each(function(i, e) {
                        $(e).removeClass('ui-selected');
                    });
                }
            }
        );
    }
}

// Initial event binding.

$(document).ready(function () {
    $('#symbol_picker').find('.symbol_preview').click(function(event) {
        symbol_preview = $(event.target);
        $('#symbol_picker').find('.symbol_preview').each(function(i, preview) {
            $(preview).removeClass('selected-symbol-preview');
        })
        symbol_preview.addClass('selected-symbol-preview');

        toolMode = $('input[name=tool_switching]:checked').val();
        if (toolMode == 'select') {
            $('.edition_grid').find('.ui-selected').each(function(i, cell) {
                symbol = getSelectedSymbol();
                setCellSymbol($(cell), symbol);
            });
        }
    });

    $('.edition_grid').each(function(i, jqGrid) {
        setUpEditionGridListeners($(jqGrid));
    });

    $('.load_task').on('change', function(event) {
        loadTaskFromFile(event);
    });

    $('.load_task').on('click', function(event) {
      event.target.value = "";
    });

    $('input[type=radio][name=tool_switching]').change(function() {
        initializeSelectable();
    });
    
    $('input[type=text][name=size]').on('keydown', function(event) {
        if (event.keyCode == 13) {
            resizeOutputGrid();
        }
    });

    $('body').keydown(function(event) {
        // Copy and paste functionality.
        if (event.which == 67) {
            // Press C

            selected = $('.ui-selected');
            if (selected.length == 0) {
                return;
            }

            COPY_PASTE_DATA = [];
            for (var i = 0; i < selected.length; i ++) {
                x = parseInt($(selected[i]).attr('x'));
                y = parseInt($(selected[i]).attr('y'));
                symbol = parseInt($(selected[i]).attr('symbol'));
                COPY_PASTE_DATA.push([x, y, symbol]);
            }
            infoMsg('Cells copied! Select a target cell and press V to paste at location.');

        }
        if (event.which == 86) {
            // Press P
            if (COPY_PASTE_DATA.length == 0) {
                errorMsg('No data to paste.');
                return;
            }
            selected = $('.edition_grid').find('.ui-selected');
            if (selected.length == 0) {
                errorMsg('Select a target cell on the output grid.');
                return;
            }

            jqGrid = $(selected.parent().parent()[0]);

            if (selected.length == 1) {
                targetx = parseInt(selected.attr('x'));
                targety = parseInt(selected.attr('y'));

                xs = new Array();
                ys = new Array();
                symbols = new Array();

                for (var i = 0; i < COPY_PASTE_DATA.length; i ++) {
                    xs.push(COPY_PASTE_DATA[i][0]);
                    ys.push(COPY_PASTE_DATA[i][1]);
                    symbols.push(COPY_PASTE_DATA[i][2]);
                }

                minx = Math.min(...xs);
                miny = Math.min(...ys);
                for (var i = 0; i < xs.length; i ++) {
                    x = xs[i];
                    y = ys[i];
                    symbol = symbols[i];
                    newx = x - minx + targetx;
                    newy = y - miny + targety;
                    res = jqGrid.find('[x="' + newx + '"][y="' + newy + '"] ');
                    if (res.length == 1) {
                        cell = $(res[0]);
                        setCellSymbol(cell, symbol);
                    }
                }
            } else {
                errorMsg('Can only paste at a specific location; only select *one* cell as paste destination.');
            }
        }
    });
});

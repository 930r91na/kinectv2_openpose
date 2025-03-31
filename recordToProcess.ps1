# Recording Processing Script
# Automatically processes recordings and supports recovery from interruptions

# Base directories
$baseDir = "C:\Users\Georg\CLionProjects\kinectv2_openpose\cmake-build-debug"
$recordingsDir = Join-Path $baseDir "recordings"
$processedDir = Join-Path $baseDir "processed"
$logsDir = Join-Path $baseDir "recordToProcessScriptLogs"

# Ensure logs directory exists
if (-not (Test-Path $logsDir)) {
    New-Item -Path $logsDir -ItemType Directory | Out-Null
}

# Log files
$logFile = Join-Path $logsDir "processing_log.txt"
$checkpointFile = Join-Path $logsDir "checkpoint.txt"

# Change to the base directory
Set-Location $baseDir

# Function to log messages
function Write-Log {
    param([string]$message)
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $message" | Out-File -FilePath $logFile -Append
    Write-Host $message
}

# Function to process a single recording
function Process-Recording {
    param([string]$recordingName)
    
    Write-Host ""
    Write-Host "****************************************************************"
    Write-Host "Processing: $recordingName"
    Write-Host "Start time: $(Get-Date)"
    Write-Host "****************************************************************"
    Write-Host ""
    
    try {
        # Run the processing command
        & "$baseDir\kinectv2_openpose.exe" --process $recordingName
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "****************************************************************"
            Write-Host "SUCCESSFULLY COMPLETED: $recordingName"
            Write-Host "End time: $(Get-Date)"
            Write-Host "****************************************************************"
            Write-Host ""
            
            # Log the completed recording
            $logMessage = "Processed: $recordingName"
            Write-Log $logMessage
            
            # Add to checkpoint if we're in a batch process
            if ($checkpointFile) {
                $logMessage | Out-File -FilePath $checkpointFile -Append
            }
            
            return $true
        } else {
            Write-Host ""
            Write-Host "****************************************************************"
            Write-Host "ERROR: Processing failed for $recordingName with exit code $LASTEXITCODE"
            Write-Host "End time: $(Get-Date)"
            Write-Host "****************************************************************"
            Write-Host ""
            
            # Log the error
            Write-Log "ERROR: Processing failed for $recordingName with exit code $LASTEXITCODE"
            
            return $false
        }
    } catch {
        $errorMsg = $_.Exception.Message
        Write-Host "An unexpected error occurred: $errorMsg"
        Write-Log "ERROR: An unexpected error occurred while processing $recordingName`: $errorMsg"
        return $false
    }
}

# Check if a specific recording was provided as an argument
if ($args.Count -gt 0) {
    $specificRecording = $args[0]
    Write-Host "Processing specific recording: $specificRecording"
    
    # Check if it contains "test" in the name
    if ($specificRecording -match "test") {
        Write-Host "Skipping test recording: $specificRecording"
    } else {
        Process-Recording $specificRecording
    }
    
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit
}

# Check if we're resuming from a previous session
$resumeMode = $false
if (Test-Path $checkpointFile) {
    Write-Host "Previous unfinished session detected."
    $resume = Read-Host "Do you want to resume from where you left off? (Y/N)"
    
    if ($resume -eq "Y" -or $resume -eq "y") {
        $resumeMode = $true
        Write-Host "Resuming previous session..."
    }
}

# Find recordings that haven't been processed yet
Write-Host "Scanning for recordings to process..."
$recordingsToProcess = @()

$allRecordings = Get-ChildItem -Path $recordingsDir -Directory

foreach ($recording in $allRecordings) {
    $dirName = $recording.Name
    
    # Skip if it contains "test" in the name
    if ($dirName -notmatch "test") {
        # Check if it exists in the processed folder
        if (-not (Test-Path (Join-Path $processedDir $dirName))) {
            # If resuming, check if this recording was already processed in the checkpoint
            $alreadyProcessed = $false
            
            if ($resumeMode) {
                $checkpointContent = Get-Content $checkpointFile -ErrorAction SilentlyContinue
                
                if ($checkpointContent -match "Processed: $dirName") {
                    $alreadyProcessed = $true
                }
            }
            
            if (-not $alreadyProcessed) {
                Write-Host "Found unprocessed recording: $dirName"
                $recordingsToProcess += $dirName
            } else {
                Write-Host "Skipping already processed recording: $dirName"
            }
        }
    }
}

# Check if we found any recordings to process
if ($recordingsToProcess.Count -eq 0) {
    Write-Host "No recordings found to process."
    
    if ($resumeMode -and (Test-Path $checkpointFile)) {
        Write-Host "All recordings from previous session are now complete."
        Remove-Item $checkpointFile -Force
    }
    
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit
}

Write-Host ""
Write-Host "Found $($recordingsToProcess.Count) recordings to process."
Write-Host ""

# Ask for confirmation
$confirm = Read-Host "Do you want to process all $($recordingsToProcess.Count) recordings? (Y/N)"

if ($confirm -ne "Y" -and $confirm -ne "y") {
    Write-Host "Operation cancelled."
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit
}

# Create/clear checkpoint file if not resuming
if (-not $resumeMode) {
    if (Test-Path $checkpointFile) {
        Remove-Item $checkpointFile -Force
    }
    New-Item -Path $checkpointFile -ItemType File -Force | Out-Null
}

# Process each recording in sequence
Write-Host ""
Write-Host "Starting to process recordings..."
Write-Host "The script can be safely interrupted and resumed later."
Write-Host ""

$current = 0
foreach ($recording in $recordingsToProcess) {
    $current++
    
    Write-Host ""
    Write-Host "****************************************************************"
    Write-Host "Processing [$current/$($recordingsToProcess.Count)]: $recording"
    Write-Host "Start time: $(Get-Date)"
    Write-Host "****************************************************************"
    Write-Host ""
    
    Process-Recording $recording
    
    Write-Host "Progress: $current/$($recordingsToProcess.Count)"
    Write-Host ""
}

Write-Host ""
Write-Host "****************************************************************"
Write-Host "All selected recordings have been processed!"
Write-Host "****************************************************************"
Write-Host ""

# Cleanup checkpoint file when all done
if (Test-Path $checkpointFile) {
    Remove-Item $checkpointFile -Force
}

Write-Host "A complete processing log is available at: $logFile"
Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")